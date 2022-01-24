import os
import pandas as pd


def analysis(table):
    table['dif'] = table['RefinePoly limit'] - table['DeepPoly limit']
    filt = table[table['dif'] > 0]
    num_imp = filt['dif'].count()
    per_impr = (filt['dif']/filt['DeepPoly limit']*100).mean()
    output = [num_imp, per_impr]
    return output

def main():
    dir_path = "raw/causality_test"
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    header = ["Model", "Num Improve", "Average Improve"]
    body = []
    for filename in onlyfiles:
        table = pd.read_csv(os.path.join(dir_path, filename))
        t_filename, _ = os.path.splitext(filename)
        analyize_detail = analysis(table)
        analyize_detail.insert(0, t_filename)
        body.append(tuple(analyize_detail))
    out_df = pd.DataFrame(body, columns=header)
    print(out_df)
    out_df.to_excel("temp.xlsx", sheet_name="Sheet1")
main()