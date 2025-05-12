import joblib

def inspect_pickle(file_path):
    obj = joblib.load(file_path)
    print(f"Inspecting {file_path}:")
    print(f"Type: {type(obj)}")
    if isinstance(obj, dict):
        print("Keys:", list(obj.keys()))
    elif isinstance(obj, (list, tuple)):
        print("Length:", len(obj))
        for i, item in enumerate(obj):
            print(f"Item {i} type: {type(item)}")
    else:
        print("Object:", obj)

if __name__ == "__main__":
    inspect_pickle('models/modele_classification_produit.pkl')
    print("\n")
    inspect_pickle('models/modele_prix_unitaire.pkl')
