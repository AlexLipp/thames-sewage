import sewage as swg

alerts = swg.get_all_discharge_alerts()
alerts.sort_values(by="DateTime", ascending=True, inplace=True)
events_df = swg.alerts_to_events_df(alerts) 
events_dict = events_df.to_dict()
swg.save_json(events_dict,"output_dir/discharge_events.json")