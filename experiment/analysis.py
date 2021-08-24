import os
import sys
import csv
import pandas as pd



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


def main():
    dir_path = "raw"
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    header = ["Model", "Verify Safe with deeppoly", "Verify Safe with refinedeeppoly", "Verify Unsafe", "Verify Unknow",
            "Total test", "Average deepoly time", "Average refinepolytime", "Avarage Unknow","Detail Success Refine", "Detail Fail Refine"]
    body = []
    for filename in onlyfiles:
            table = pd.read_csv(os.path.join(dir_path, filename))
            t_filename, _ = os.path.splitext(filename)
            analyize_detail = analysis(table)
            analyize_detail.insert(0, t_filename)
            body.append(tuple(analyize_detail))
            #print(analyize_detail)
    out_df = pd.DataFrame(body, columns=header)
    print(out_df)
    out_df.to_excel("analysis.xlsx", sheet_name="Sheet1")
main()