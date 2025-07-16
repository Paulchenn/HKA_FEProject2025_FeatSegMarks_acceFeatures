# Tool um die Inhalte von tuned_G_119 zu untersuchen
# Gibt Überblick über die Architektur der .pth-Dateien, die gespeichert wurden
# So kann eine passende Modellklasse erstellen

import torch

# Pfad zur .pth-Datei im Container
pth_path = "/app/code/tuned_G_119.pth"

print(f"Lade Modell-Datei: {pth_path}")
try:
    state = torch.load(pth_path, map_location="cpu")
except Exception as e:
    print(f"Fehler beim Laden der Datei: {e}")
    exit(1)

# Wenn state_dict verschachtelt ist, extrahieren
if isinstance(state, dict):
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
else:
    print("Unerwartetes Format (kein Dictionary).")
    exit(1)

# Zeige Key-Typen und einen Auszug
print("\nErfolgreich geladen. Schlüssel im Modell:")
keys = list(state_dict.keys())
print(f"Anzahl Parameter: {len(keys)}")
print("Beispiel-Keys:\n")

for i, k in enumerate(keys):  
    print(f"{i+1:2d}: {k}")

# Optional: Infos zur Gewichtsdimension
print("\nBeispiel-Dimensionen:")
for k in keys:
    print(f"{k:40s} → {tuple(state_dict[k].shape)}")
