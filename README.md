# Smart Waste Routing

Pipeline simple pour:
- simuler l'etat de remplissage des poubelles,
- predire le remplissage J+1/J+3 par poubelle (ML time-series),
- optimiser l'ordre de collecte des poubelles critiques,
- generer une carte HTML,
- visualiser les resultats dans un dashboard Streamlit.

## Lancer le systeme

```bash
pip install -r requirements.txt
python main.py
```

Sorties generees:
- `data/bins_with_status.csv`
- `data/fill_history.csv`
- `data/fill_predictions.csv`
- `data/optimized_route.csv`
- `maps/smart_waste_map.html`

## Lancer le dashboard

```bash
streamlit run dashboard/app.py
```
