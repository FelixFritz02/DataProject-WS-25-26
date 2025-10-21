from ucimlrepo import fetch_ucirepo

class HeartFailureDataset:
    def __init__(self):
        # Dataset abrufen
        heart_failure_clinical_records = fetch_ucirepo(id=519)
        
        # Daten als Attribute speichern
        self.X = heart_failure_clinical_records.data.features
        self.y = heart_failure_clinical_records.data.targets
    
    def get_features(self):
        """Gibt die Features zurück."""
        return self.X

    def get_targets(self):
        """Gibt die Zielvariable zurück."""
        return self.y
  
