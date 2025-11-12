from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset, ClassificationPreset
from evidently.ui.workspace import Snapshot

class MonitorCore:
    def __init__(self, ws, project_id, data_loader):
        self.ws = ws
        self.project_id = project_id
        self.loader = data_loader

    def generate_reports(self):
        ref_df = self.loader.load_reference_data()
        live_df = self.loader.load_live_data()

        if live_df.empty:
            print("No live data available for monitoring")
            return
        
        ref_ds, live_ds = self.loader.to_datasets(ref_df, live_df)

        # save datasets in evidently
        self.ws.add_dataset(
            dataset=ref_ds,
            name="reference_dataset",
            project_id=self.project_id,
            description="Reference dataset during training"
        )
        self.ws.add_dataset(
            dataset=live_ds,
            name="live_dataset",
            project_id=self.project_id,
            description="Live dataset in real-time"
        )

        data_quality = Report([ DataSummaryPreset() ], include_tests=True)
        data_drift = Report([ DataDriftPreset() ], include_tests=True)
        prediction_drift = Report([ ValueDrift(column='prediction') ], include_tests=True)
        classy_report = Report([ ClassificationPreset() ], include_tests=True)

        # process reports
        reports = {
            "data_quality": data_quality,
            "data_drift": data_drift,
            "prediction_drift": prediction_drift,
            "classification": classy_report,
        }

        results = {}

        for name, report in reports.items():
            print(f"Running {name} report....")
            eval = report.run(current_data=live_ds, reference_data=ref_ds)
            self.ws.add_run(self.project_id, eval)
            results[name] = eval

        return results
    
    