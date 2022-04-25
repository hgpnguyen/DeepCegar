import os
from numpy import average
import pandas as pd
import csv
import ast


def analysis(table):
    table.loc[table["Total number of task"]==1, "Verify result"] = "Verified_Result.DeepSafe"
    table["vers"] = table["Verify result"].astype("category")
    #print(table.vers.cat.categories)
    table.vers = table.vers.cat.set_categories(['Verified_Result.DeepSafe', 'Verified_Result.Safe',
       'Verified_Result.UnSafe', 'Verified_Result.Unknow'])
    total_rows = table.shape[0]
    group = table.groupby("vers")
    size_each_group = list(group.size())
    output = size_each_group
    output.append(total_rows)
    
    time_avarage = group.time.mean()
    #print(time_avarage["Verified_Result.Safe"])
    output = output + [time_avarage["Verified_Result.DeepSafe"], time_avarage["Verified_Result.Safe"], time_avarage["Verified_Result.Unknow"]]

    safe = table.loc[table.vers=='Verified_Result.Safe', ["Total number of task", "time"]]
    detail_safe = list(zip(list(safe["Total number of task"]), list(safe["time"])))
    output.append(detail_safe)
    #print(detail_safe)
    unknow = table.loc[table.vers=='Verified_Result.Unknow', ["Total number of task", "time"]]
    detail_unknow = list(zip(list(unknow["Total number of task"]), list(unknow["time"])))
    output.append(detail_unknow)
    #print(detail_unknow)
    #print(output)
    return output

def get_k(filename):
    idx = filename.find('_k')
    return int(filename[idx+2])

def get_time(filename, idx):
    time_dict = {}
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            time_lst = ast.literal_eval(row[idx])
            time_dict[row[0]] = time_lst[-1] if time_lst else None
    return time_dict

def compute_time():
    dir_path = "raw/causality_test"
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    dir_path2 = "raw/limit_test"
    onlyfiles2 = [f for f in os.listdir(dir_path2) if os.path.isfile(os.path.join(dir_path2, f))]
    dir_path3 = "raw/deeppoly_test"
    onlyfiles3 = [f for f in os.listdir(dir_path3) if os.path.isfile(os.path.join(dir_path3, f))]
    header = ["Causal file", "Gradient file", "DeepPoly file", "Causal time", "Gradient time", "DeepPoly time"]
    body = []
    for idx in range(len(onlyfiles)):
        causal_time = get_time(os.path.join(dir_path, onlyfiles[idx]), 3)
        gradient_time = get_time(os.path.join(dir_path2, onlyfiles2[idx]), 3)
        deepoly_time = get_time(os.path.join(dir_path3, onlyfiles3[idx]), 2)
        causal_filename, _ = os.path.splitext(onlyfiles[idx])
        gradient_filename, _ = os.path.splitext(onlyfiles2[idx])
        deepoly_filename, _ = os.path.splitext(onlyfiles3[idx])
        for key in causal_time:
            if causal_time[key] is None:
                causal_time[key] = deepoly_time[key]
        for key in gradient_time:
            if gradient_time[key] is None:
                gradient_time[key] = deepoly_time[key]
        if idx == 1:
            print(causal_time)
            print(gradient_time)
            print(deepoly_time)
        causal_sum = sum(causal_time.values())
        gradient_sum = sum(gradient_time.values())
        deepoly_sum = sum(deepoly_time.values())
        time_tup = (causal_filename, gradient_filename, deepoly_filename, causal_sum, gradient_sum, deepoly_sum)
        body.append(time_tup)
    out_df = pd.DataFrame(body, columns=header)
    #print(out_df)
    out_df.to_excel("temp.xlsx", sheet_name="Sheet1")
    
    

def main():
    dir_path = "raw/free_method"
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    header = ["Model", "k", "Verify Safe with deeppoly", "Verify Safe with refinedeeppoly", "Verify Unsafe", "Verify Unknow",
            "Total test", "Average deepoly time", "Average refinepolytime", "Avarage Unknow","Detail Success Refine", "Detail Fail Refine"]
    body = []
    for filename in onlyfiles:
        table = pd.read_csv(os.path.join(dir_path, filename))
        t_filename, _ = os.path.splitext(filename)
        analyize_detail = analysis(table)
        analyize_detail.insert(0, get_k(t_filename))
        analyize_detail.insert(0, t_filename)
        body.append(tuple(analyize_detail))
        #print(analyize_detail)
    out_df = pd.DataFrame(body, columns=header)
    print(out_df)
    out_df.to_excel("temp.xlsx", sheet_name="Sheet1")
compute_time()