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

## Mode simulation continue (temps reel)

```bash
python main.py --live --ticks 300 --tick-seconds 3 --sim-hours-per-tick 0.5 --speed-factor 6
```

- `--live`: lance une simulation qui met a jour le remplissage en boucle dans un seul run.
- `--tick-seconds`: frequence de mise a jour reelle.
- `--sim-hours-per-tick`: temps simule ajoute a chaque tick.
- `--speed-factor`: multiplicateur de vitesse de remplissage.
- En mode `--live`, le systeme ne calcule pas automatiquement la trajectoire.
- La trajectoire est calculee uniquement via le bouton `Voir la trajectoire` dans le dashboard.
- Le vidage se fait via le bouton `Lancer collecte` dans le dashboard.

Sorties generees:
- `data/bins_with_status.csv`
- `data/bins_state.csv`
- `data/fill_events.csv`
- `data/fill_history.csv`
- `data/fill_predictions.csv`
- `data/optimized_route.csv`
- `maps/smart_waste_map.html`

## Lancer le dashboard

```bash
streamlit run dashboard/app.py
```
