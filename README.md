# Lifestyle and Health Monitoring

The first prototype connects sample meal and step capture to a local API, persistent storage, and a daily dashboard. The base server runs with Python 3.9 or later and uses the standard library. Photo recognition additionally requires the packages in requirements-photo.txt, using a Python version supported by the selected PyTorch and Torchvision releases.

## Start in VS Code

1. Open this folder in VS Code.
2. Select your Python interpreter with **Python: Select Interpreter**.
3. In **Run and Debug**, choose **Run local health prototype** and press **F5**. You can also run `python server.py` in the integrated terminal.
4. Open [the local dashboard](http://127.0.0.1:8000/).
5. Enter a sample meal of 450 kcal and a sample activity interval of 1,500 steps on the same date. The daily view shows both totals.

Use synthetic data for this prototype. It binds to the loopback address and uses plaintext local SQLite and browser storage. It is a development tool for one browser and one test participant. Keep step intervals distinct since the prototype sums each submitted step record.

## Try an interrupted sync

1. Open the dashboard and leave that tab open.
2. Stop the server and save another sample meal. The browser keeps it pending.
3. Restart the server and click **Sync pending records**. The total should include the new meal once.

The dashboard page needs the server to load. The browser can queue entries from an already open page while the server is down. SQLite data lives in `.data/prototype.sqlite3`, which Git ignores.

## First integration contract

- `POST /api/v1/records` accepts JSON such as `{"id":"332daab7-ced8-4d08-8b44-90bd7f89b519","day":"2026-11-12","kind":"meal","value":450}`. For `steps`, the value is the step count for a distinct activity interval. Send the same ID and payload on retry. A reused ID with different content returns HTTP 409.
- `GET /api/v1/daily?day=2026-11-12` returns meal energy, step totals, and the saved meals for that date. A missing total returns `null`. Manual meals have `null` for food name, portion and energy density.

## Try food photos

Install the optional photo dependencies and download the pinned model once from the project folder:

```powershell
python -m pip install -r requirements-photo.txt
python food_photo.py
python server.py
```

On the dashboard, choose a JPEG, PNG or WebP image. Analysis starts automatically. Click **Analyze photo locally** to retry if needed. Select or correct a dish name. Choose an example starting portion (150, 300 or 450 g), or enter a custom weight. The buttons are editable demo amounts. Energy shortcuts of 100, 160 and 240 kcal per 100 g are also provided as example values for testing. Enter the energy per 100 g from the food label or recipe. Check the estimated energy, then confirm the meal. For example, 150 g at 240 kcal per 100 g gives 360 kcal.

The model suggests up to three whole-dish labels. Portion weight and energy density come from the user. If analysis fails or none of the labels fit, type the dish name manually. Inference runs on the local CPU without network access. Photo bytes are processed in memory. The database keeps the confirmed name, portion and energy values. Use a public sample photo or other non-sensitive demo input.

The model is [MobileNet V2 trained on Food-101](https://huggingface.co/AlexKoff88/mobilenet_v2_food101), pinned by revision and SHA-256. The class list follows the [Food-101 labels](https://huggingface.co/nateraw/food/blob/main/config.json) in the [sort order used by Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py).

- `POST /api/v1/food-photo` takes raw image bytes with the corresponding `Content-Type` header. It returns the model version and ranked suggestions with model scores.
- Photo meal saves use the existing records endpoint with an optional `photo_meal` object containing `food`, `grams` and `kcal_per_100g`. Only the confirmed meal details enter the retry queue.

## Evaluate dish recognition

After the photo setup, create `.data/food-model/evaluation.csv` with one row per photo. Paths are relative to the CSV. For the sample pizza photo in that folder, use:

```csv
photo,expected
sample-pizza.jpg,pizza
```

Run `python evaluate_food_photos.py .data/food-model/evaluation.csv`. The JSON report shows the ranked suggestions for each photo, how often the first suggestion was correct, and how often the expected dish appeared anywhere in the first three. The script uses the same local classifier as the dashboard and does not write meal records.

Use public test photos and choose the expected label from `food101_labels.txt` before running the model. Record dishes outside Food-101 and mixed plates with several correct answers separately. This evaluation checks one whole-dish label per photo.

The next integration is a SwiftUI client on a Mac with Xcode and an iPhone for HealthKit testing. Before enabling LAN access or collecting real data, add authenticated HTTPS, device enrollment, and encrypted storage, then review the collection and retention policy. Add the production database and richer time and provenance fields as the client contract develops.
