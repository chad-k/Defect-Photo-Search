"""Load packaged photorealistic reference images; never generate graphical fallbacks."""
import json
from pathlib import Path

TYPES = ['Scratch', 'Crack', 'Pitting', 'Dent', 'Stain', 'Corrosion', 'Burr', 'Molding flash', 'Burn mark', 'No visible defect']

def ensure_demo(root):
    root = Path(root)
    manifest = root / 'cases.json'
    if not manifest.exists():
        raise FileNotFoundError('The realistic demo collection is missing. Extract the demo_realistic folder from the updated app ZIP next to app.py.')
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if len(rows) != 100:
        raise ValueError('Expected 100 realistic demo reference images. Extract the complete demo_realistic folder from the updated ZIP.')
    missing = [row['image'] for row in rows if not (root / row['image']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} reference images are missing. Extract the complete demo_realistic folder.')
