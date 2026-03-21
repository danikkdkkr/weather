Guide the user through integrating a new data source into the weather forecasting pipeline.

Ask the user:
1. Which data source to add (e.g. "ERA5", "OpenMeteo", "ICON NWP", "satellite indices")
2. What new features it provides and their expected value ranges
3. Whether the features should be added as additional input channels or used to replace existing ones

Then:
- Describe the recommended fetch/cache strategy (API vs local files) for that source
- Show how to extend `process_station_df()` in `data_pipeline.py` to incorporate the new features
- Show how to update `normalize()` in `data_pipeline.py` to handle any new columns (add to skip_cols if already normalised, otherwise include in scaling)
- Remind the user that the **target columns and their order remain fixed** — new sources add input features only
- For multi-station data: the ring+segment search (`WeatherStation.find_nearby` in `stations.py`) already handles station discovery; new sources should integrate into `fetch_multi_station_data()` in `data_pipeline.py`

Follow the conventions in CLAUDE.md.
