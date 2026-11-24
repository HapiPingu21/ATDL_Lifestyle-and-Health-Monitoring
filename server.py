"""Loopback-only prototype for synthetic meal and activity records."""

import argparse
from contextlib import closing
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from food_photo import MAX_PHOTO_BYTES, PHOTO_TYPES, PhotoUnavailable, suggest_food


ROOT = Path(__file__).resolve().parent
LIMITS = {"meal": 100_000, "steps": 1_000_000}


def parse_day(value):
    parsed = date.fromisoformat(value)
    if value != parsed.isoformat():
        raise ValueError("Use YYYY-MM-DD for the date.")
    return value


def parse_record(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON record.")
    record_id = str(UUID(payload["id"]))
    day = parse_day(payload["day"])
    kind = payload["kind"]
    value = payload["value"]
    if kind not in LIMITS:
        raise ValueError("Use meal or steps as the record type.")
    if type(value) is not int or not 0 <= value <= LIMITS[kind]:
        raise ValueError(f"Use a whole number between 0 and {LIMITS[kind]}.")
    return record_id, day, kind, value


def parse_photo_meal(payload, record):
    if "photo_meal" not in payload:
        return None
    details = payload["photo_meal"]
    if record[2] != "meal" or not isinstance(details, dict):
        raise ValueError("Photo details require a meal record.")
    food, grams, density = details["food"].strip(), details["grams"], details["kcal_per_100g"]
    if not 1 <= len(food) <= 100:
        raise ValueError("Enter a food name of up to 100 characters.")
    if type(grams) is not int or not 1 <= grams <= 10_000:
        raise ValueError("Enter a portion between 1 and 10,000 grams.")
    if type(density) is not int or not 0 <= density <= 1_000:
        raise ValueError("Enter a whole number between 0 and 1,000 kcal per 100 g.")
    if record[3] != (grams * density + 50) // 100:
        raise ValueError("Meal energy does not match the portion and energy per 100 g.")
    return food, grams, density


class PrototypeServer(ThreadingHTTPServer):
    def __init__(self, port, database):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    day TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('meal', 'steps')),
                    value INTEGER NOT NULL CHECK (value >= 0)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS photo_meals (
                    record_id TEXT PRIMARY KEY REFERENCES records(id),
                    food TEXT NOT NULL,
                    grams INTEGER NOT NULL,
                    kcal_per_100g INTEGER NOT NULL
                )
            """)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, status, payload):
        self.reply(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.reply(HTTPStatus.OK, (ROOT / "dashboard.html").read_bytes(),
                       "text/html; charset=utf-8")
            return
        if url.path != "/api/v1/daily":
            self.json_reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
            return

        try:
            day = parse_day(parse_qs(url.query)["day"][0])
        except (KeyError, ValueError):
            self.json_reply(HTTPStatus.BAD_REQUEST, {"error": "Use ?day=YYYY-MM-DD."})
            return

        with closing(sqlite3.connect(self.server.database)) as db:
            db.execute("BEGIN")
            totals = dict(db.execute(
                "SELECT kind, SUM(value) FROM records WHERE day = ? GROUP BY kind",
                (day,),
            ))
            meals = db.execute("""
                SELECT records.value, photo_meals.food, photo_meals.grams,
                       photo_meals.kcal_per_100g
                FROM records
                LEFT JOIN photo_meals ON photo_meals.record_id = records.id
                WHERE records.day = ? AND records.kind = 'meal'
                ORDER BY records.rowid
            """, (day,)).fetchall()
        self.json_reply(HTTPStatus.OK, {
            "day": day, "meal_kcal": totals.get("meal"), "steps": totals.get("steps"),
            "meals": [
                {"kcal": kcal, "food": food, "grams": grams, "kcal_per_100g": density}
                for kcal, food, grams, density in meals
            ],
        })

    def do_POST(self):
        if self.path == "/api/v1/food-photo":
            self.analyze_photo()
            return
        if self.path != "/api/v1/records":
            self.json_reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use JSON."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Use a JSON body of at most 4096 bytes.")
            payload = json.loads(self.rfile.read(size))
            record = parse_record(payload)
            details = parse_photo_meal(payload, record)
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            self.json_reply(HTTPStatus.BAD_REQUEST, {"error": f"Invalid record. {error}"})
            return

        with closing(sqlite3.connect(self.server.database)) as db, db:
            inserted = db.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                record,
            ).rowcount
            if inserted and details is not None:
                db.execute("INSERT INTO photo_meals VALUES (?, ?, ?, ?)",
                           (record[0], *details))
            existing = db.execute(
                "SELECT id, day, kind, value FROM records WHERE id = ?", (record[0],)
            ).fetchone()
            existing_details = db.execute(
                "SELECT food, grams, kcal_per_100g FROM photo_meals WHERE record_id = ?",
                (record[0],),
            ).fetchone()

        if existing != record or existing_details != details:
            self.json_reply(HTTPStatus.CONFLICT, {"error": "Record ID already in use."})
            return
        self.json_reply(HTTPStatus.CREATED if inserted else HTTPStatus.OK,
                        {"id": record[0], "status": "saved" if inserted else "duplicate"})

    def analyze_photo(self):
        content_type = self.headers.get_content_type()
        if content_type not in PHOTO_TYPES:
            self.json_reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                            {"error": "Choose a JPEG, PNG, or WebP photo."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_PHOTO_BYTES:
                raise ValueError("Choose a photo no larger than 8 MiB.")
            self.connection.settimeout(15)
            photo = self.rfile.read(size)
            if len(photo) != size:
                raise ValueError("The photo upload was incomplete.")
            result = suggest_food(photo, content_type)
        except TimeoutError:
            self.json_reply(HTTPStatus.REQUEST_TIMEOUT, {"error": "The upload timed out."})
        except ValueError as error:
            self.json_reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except PhotoUnavailable as error:
            self.json_reply(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
        except (OSError, RuntimeError):
            self.json_reply(HTTPStatus.SERVICE_UNAVAILABLE,
                            {"error": "The local photo model could not run."})
        else:
            self.json_reply(HTTPStatus.OK, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--database", type=Path, default=ROOT / ".data" / "prototype.sqlite3")
    args = parser.parse_args()
    with PrototypeServer(args.port, args.database) as server:
        print(f"Open http://127.0.0.1:{server.server_port} in your browser.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
