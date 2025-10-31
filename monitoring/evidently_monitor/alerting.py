from .prometheus_metric import (
    DATA_DRIFT_DETECTED, DRIFTED_COLUMNS, PREDICTION_DRIFT_DETECTED,
    MODEL_ACCURACY_CURRENT, MODEL_ACCURACY_REFERENCE, ALERT_COUNT
)

class AlertEngine:
    def __init__(self, output_path):
        self.output_path = output_path

    def process_reports(self, reports, timestamp):
        alerts = []

        # ---- Data drift alerts ----
        dd = reports.get('data_drift')
        print('dd: ', dd)
        drift_detected = False
        drifted_columns_count = 0

        if dd:
            drift_result = dd.dict()
            tests = drift_result.get('tests', [])
            for test in tests:
                status = str(test.get("status", "")).upper()
                name = test.get("column", "Unnamed Test")
                desc = test.get("description", "") 
                if status == "TESTSTATUS.FAIL":
                    drift_detected = True
                    alerts.append(f"{name} — {desc}")
            
            metrics = drift_result.get('metrics', [])
            for m in metrics:
                if "Share_of_drifted_columns" in m.get('metric', ''):
                    drifted_columns_count = m['result'].get("number_of_drifted_columns", 0)
                    break
        DATA_DRIFT_DETECTED.set(1 if drift_detected else 0)
        DRIFTED_COLUMNS.set(drifted_columns_count)

        # --- Value Drift (Prediction drift) ---
        vd = reports.get('prediciton_drift')
        prediction_drift_detected = False

        if vd:
            vd_result = vd.dict()
            tests = vd_result.get('tests', [])
            for test in tests:
                status = str(test.get("status", "")).upper()
                if status == "FAIL":
                    prediction_drift_detected = True
                    desc = test.get("description", "")
                    alerts.append(f"Prediction drift detected: {desc}")
        PREDICTION_DRIFT_DETECTED.set(1 if prediction_drift_detected else 0)


        # --- classification performance alerts ---
        classy_report = reports.get('classification')
        acc_curr = acc_ref = None

        if classy_report:
            classy_dict = classy_report.dict()
            metrics = classy_dict.get("metrics", [])
            for metric in metrics:
                result = metric.get("result", {})
                if isinstance(result, dict):
                    acc_curr = result.get("current", {}).get("accuracy")
                    acc_ref = result.get("reference", {}).get("accuracy")
                    if acc_curr and acc_ref and acc_curr < acc_ref * 0.9:
                        alerts.append(f"Model accuracy dropped significantly ({acc_ref:.3f} → {acc_curr:.3f})")
                        break
        if acc_curr is not None:
            MODEL_ACCURACY_CURRENT.set(acc_curr)
        if acc_ref is not None:
            MODEL_ACCURACY_REFERENCE.set(acc_ref)


        # Save alerts to a file
        if alerts:
            ALERT_COUNT.inc(len(alerts))
            alerts_path = f"{self.output_path}/alerts_{timestamp}.txt"
            
            with open(alerts_path, "w") as f:
                f.write("\n".join(alerts))
            print(f"⚠️ Alerts generated and saved to alerts_{timestamp}.txt")
        
        else:
            print("✅ No alerts generated.")
