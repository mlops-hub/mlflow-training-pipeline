import joblib
import pandas as pd


MODEL_PATH = '../../models/best_model.pkl'
FEATURE_PATH = '../../feature_store/feature_names.pkl'
SCALER = "../../utility/scaler.pkl"
PREPROCESSED_DATASET = '../../datasets/preprocess/preprocessing_df.csv'


# Get the features for prediction
def create_animal(animal_name):
    print("Enter features here: ")
    get_hair = input("hair(0 or 1): ")
    get_eggs = input("eggs(0 or 1): ")
    get_milk = input("milk(0 or 1): ")
    get_predator = input("predator(0 or 1: ")
    get_toothed = input("toothed(0 or 1): ")
    get_backbone = input("backbone(0 or 1): ")
    get_breathes = input("breathes(0 or 1): ")
    get_venomous = input("venomous(0 or 1): ")
    get_legs = input("legs(0,2,4,5,6,8): ")
    get_tail = input("tail(0 or 1): ")
    can_fly = input("can_fly(0 or 1): ")
    can_swim = input("can_swim(0 or 1): ")
    is_domestic_pet = input("is_domestic_pet(0 or 1): ")
    features_dict = {
        'animal_name':  animal_name, 'hair': int(get_hair), 'eggs': int(get_eggs), 'milk': int(get_milk), 
        'predator': int(get_predator), 'toothed': int(get_toothed),
        'backbone': int(get_backbone), 'breathes': int(get_breathes), 
        'can_fly': int(can_fly), 'can_swim': int(can_swim), 'is_domestic_pet': int(is_domestic_pet),
        'venomous': int(get_venomous), 'legs': int(get_legs), 'tail': int(get_tail), 
    }
    # print(features_dict)
    return features_dict



def predict_animal(animal_name):
    #  load model, scaler, feature_name, df
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_PATH)
        scaler = joblib.load(SCALER)
        df = pd.read_csv(PREPROCESSED_DATASET)
        print("✅ model loaded")
    except Exception as e:
        print(e)

    # load dataset to get animal_name features
    feature = df[df['animal_name'] == animal_name]

    if feature.empty:
        print("This animal is not found in dataset.")
        feature = create_animal(animal_name)
        feature_df = pd.DataFrame([feature])[feature_names]
        scaled_feature = scaler.transform(feature_df.drop(columns=['animal_name']))
        input = scaled_feature

    else: 
        input = feature.drop(columns=['animal_name'])

    y_result = model.predict(input)[0]
    print(f"\nAnimal Type for {animal_name}: {y_result}\n")


if __name__ == "__main__":
    while True:
        animal_name = input("Enter animal name: ")
        if animal_name.lower() in ['exit', 'quit']:
            print("👋🏻 Bye")
            break
        predict_animal(animal_name)

