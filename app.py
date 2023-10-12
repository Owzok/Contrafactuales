from flask import Flask, render_template, request, redirect
import pandas as pd
import dice_ml
import joblib
import random
import json

clf = joblib.load("Fixed_model.pkl")
df = pd.read_csv("Data_limpia2.csv")

d = dice_ml.Data(dataframe=df, continuous_features=['PONDERADO', 'IDH', 'HRS_FALTA_RATIO', 'POR_POBREZA', 'POR_POBREZA_EXTREMA', 'Tasa_Analfabetismo', 'Tasa_Desempleo2019'], outcome_name='ESTADO')

m = dice_ml.Model(model_type='classifier', model=clf, backend='sklearn')

dice = dice_ml.Dice(d, m)

dropouts_list_index = df[df.ESTADO == -1].index.to_list()
df.drop(["ESTADO"], axis=1, inplace=True)

app = Flask(__name__)

def custom_counterfactuals(x):
    semestre_actual = x.SEM_CURSADOS.values[0]
    ponderado = x.PONDERADO.values[0]
    faltas = x.HRS_FALTA_RATIO.values[0]
    aprobo = x.APROBO.values[0]
    creditos = x.CREDITOS.values[0]
    contrafactuales = dice.generate_counterfactuals(x, 
                total_CFs=3, desired_class=1,
                features_to_vary=['PONDERADO', 'HRS_FALTA_RATIO', 'ESTADO_CIVIL_D', 'BECA_VIGENTE'],
                permitted_range={
                    "PONDERADO": [ponderado, 20],
                    "HRS_FALTA_RATIO": [0, faltas]
                })
    return contrafactuales

@app.route("/random_counterfactual", methods=["GET", "POST"])
def generate_random_counterfactual():
    while True:
        try:
            random_index = random.choice(dropouts_list_index)
            random_query = df.iloc[random_index:random_index+1]
            counterfactuals = custom_counterfactuals(random_query)

            formatted_json = json.loads(counterfactuals.to_json())
            print(json.loads(counterfactuals.to_json()))
            test_data = pd.DataFrame(formatted_json["test_data"][0], columns=formatted_json["feature_names_including_target"])
            cfs_list = pd.DataFrame(formatted_json["cfs_list"][0], columns=formatted_json["feature_names_including_target"])
            cfs_list.columns = cfs_list.columns.str.replace(' ', '_')

            print(cfs_list)
            return render_template("result.html", counterfactuals=formatted_json, test_data=test_data, cfs_list=cfs_list)
        except Exception as e:
            print(e)

@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None 
    if request.method == "POST":
        try:
            if request.form["submit"] == "Choose Randomly":
                return redirect("/random_counterfactual")
            else:
                input_data = df.iloc[0:1].copy()
                input_data['SEXO'] = int(request.form["SEXO"])
                input_data['SEM_CURSADOS'] = int(request.form["SEM_CURSADOS"])
                input_data['CREDITOS'] = int(request.form["CREDITOS"])
                input_data["APROBO"] = int(request.form["APROBO"])
                input_data["PONDERADO"] = float(request.form["PONDERADO"])
                input_data["BECA_VIGENTE"] = int(request.form["BECA_VIGENTE"])
                input_data["ESTADO_CIVIL_D"] = int(request.form["ESTADO_CIVIL_D"])
                input_data["ESTADO_CIVIL_S"] = int(request.form["ESTADO_CIVIL_S"])
                input_data["CRED_X_SEM"] = float(request.form["CRED_X_SEM"])
                input_data["HRS_FALTA_RATIO"] = float(request.form["HRS_FALTA_RATIO"])
                input_data["EDAD"] = float(request.form["EDAD"])
                input_data["IDH"] = float(request.form["IDH"])
                input_data["POR_POBREZA"] = float(request.form["POR_POBREZA"])
                input_data["POR_POBREZA_EXTREMA"] = float(request.form["POR_POBREZA_EXTREMA"])
                input_data["Tasa_Analfabetismo"] = float(request.form["Tasa_Analfabetismo"])
                input_data["Tasa_Desempleo2019"] = float(request.form["Tasa_Desempleo2019"])
                #print(input_data)
                query = pd.DataFrame(input_data, index=[0])
                counterfactuals = custom_counterfactuals(query)

                formatted_json = json.loads(counterfactuals.to_json())

                test_data = pd.DataFrame(formatted_json["test_data"][0], columns=formatted_json["feature_names_including_target"])
                cfs_list = pd.DataFrame(formatted_json["cfs_list"][0], columns=formatted_json["feature_names_including_target"])
                cfs_list.columns = cfs_list.columns.str.replace(' ', '_')

                return render_template("result.html", counterfactuals=counterfactuals, test_data=test_data, cfs_list=cfs_list)
        except ValueError as e:
            error_message = str(e)
            print(e)
    
    return render_template("form.html", error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True)