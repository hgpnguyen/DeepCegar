import os
import pandas as pd


def analysis(table):
    table['dif'] = table['RefinePoly limit'] - table['DeepPoly limit']
    filt = table[table['dif'] > 0]
    num_imp = filt['dif'].count()
    per_impr = (filt['dif']/filt['DeepPoly limit']*100).mean()
    output = [num_imp, per_impr]
    return output

def analysis2(table, table2, idx):
    table['dif'] = table2['RefinePoly limit'] - table['RefinePoly limit']
    table2['dif'] = table['RefinePoly limit'] - table2['RefinePoly limit']
    filt = table[table['dif'] > 0]
    filt2 = table2[table2['dif'] > 0]
    num_imp = filt['dif'].count()
    num_imp2 = filt2['dif'].count()
    #if idx == 39:
        #filt.to_excel("temp2.xlsx", sheet_name="Sheet1")
    per_impr = (filt['dif']/filt['RefinePoly limit']*100).mean()
    per_impr2 = (filt2['dif']/filt2['RefinePoly limit']*100).mean()
    output = [num_imp, num_imp2, per_impr, per_impr2]
    return output

def main():
    dir_path = "raw/causality_test"
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    dir_path2 = "raw/limit_test"
    onlyfiles2 = [f for f in os.listdir(dir_path2) if os.path.isfile(os.path.join(dir_path2, f))]
    header = ["Model", "Num Gradine Improve", "Num Causality Improve", "Average Gradient Improve", "Average Causality Improve"]
    body = []
    for idx in range(len(onlyfiles)):
        table = pd.read_csv(os.path.join(dir_path, onlyfiles[idx]))
        t_filename, _ = os.path.splitext(onlyfiles[idx])
        table2 = pd.read_csv(os.path.join(dir_path2, onlyfiles2[idx]))
        analyize_detail = analysis2(table, table2, idx)
        analyize_detail.insert(0, t_filename)
        body.append(tuple(analyize_detail))
    out_df = pd.DataFrame(body, columns=header)
    print(out_df)
    out_df.to_excel("temp.xlsx", sheet_name="Sheet1")
main()