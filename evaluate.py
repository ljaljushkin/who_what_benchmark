from transformers import AutoTokenizer
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path
import json

tokenizer_path = "/home/devuser/nlyalyus/projects/lm-evaluation-harness/cache/stable-zephyr-3b-dpo/fp16"
prefix = 'generations_'
ref_name = "fp16"
cmp_names = [
    'int4_g64_nozp_r80_hawq_OUT2',
    'int4_g64_nozp_r80',
    'int4_g64_nozp_r80_hawq_IN',
    'int8',
    'fp16'
]

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    revision=None,
    trust_remote_code=True,
    use_auth_token=True,
    truncation_side="left",
    padding_side="right",  # padding on the right is needed to cut off padding in `complete_code`
)
DEBUG = False

results_file = Path('results.json')
all_results = []
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
else:
    ref_file = prefix + ref_name + '.csv'
    ref_data = pd.read_csv(ref_file)
    for cmp_name in cmp_names:
        cmp_file = prefix + cmp_name + '.csv'
        cmp_data = pd.read_csv(cmp_file)

        # d = dict(zip(ref_data.questions,ref_data.answers))
        # TODO: check that corresponds to the same question

        fdt_list = []
        sdt_list = []
        sdtr_list = []

        DEBUG = False
        fdt_max = 0

        # NOTE: a - reference answers, b - answers to evaluate
        for a_answer, b_answer in zip(ref_data.answers, cmp_data.answers):
            a_indexes = tokenizer.encode(a_answer, return_tensors="pt").squeeze().tolist()
            b_indexes = tokenizer.encode(b_answer, return_tensors="pt").squeeze().tolist()
            fdt_max += len(a_indexes)

            matcher = SequenceMatcher(None, a_indexes, b_indexes)
            blocks = matcher.get_matching_blocks()
            a, b, size = blocks[0]
            fdt = 0
            if a == 0 and b == 0:
                fdt = blocks[0].size
            fdt_list.append(fdt)
            num_matched = sum(block.size for block in blocks)
            sdt = len(b_indexes) - num_matched
            sdtr = len(a_indexes) - num_matched

            sdtr_list.append(sdtr)
            sdt_list.append(sdt)
            if DEBUG:
                results = {
                    'FDT': fdt,
                    'SDT': sdt,
                    'SDTR norm': sdtr,
                }
                print(json.dumps(results, indent=4))
                print(blocks)
                for block in blocks:
                    a, b, size = block
                    matched = a_indexes[a : a + size + 1]
                    print(matched)
                    print(tokenizer.decode(matched))
                    matched = b_indexes[b : b + size + 1]
                    print(matched)
                    print(tokenizer.decode(matched))

        num_answers = len(fdt_list)
        fdt_max = fdt_max / num_answers
        fdt_score = sum(fdt_list) / num_answers
        sdt_score = sum(sdt_list) / num_answers
        sdtr_score = sum(sdtr_list) / num_answers

        fdt_norm = fdt_score / fdt_max
        sdtr_norm = sdtr_score/ fdt_max
        results = {
            'mode': cmp_name,
            'FDT': fdt_score,
            'FDT norm': fdt_norm,
            'SDT': sdt_score,
            'SDTR norm': sdtr_norm,
        }
        all_results.append(results)

        print(f"\n\n\nScores for model: {Path(cmp_file).name}\n")
        print(json.dumps(results, indent=4))

    print(json.dumps(all_results, indent=4))
    with open(results_file, 'w') as f:
        json.dump(all_results, f)


df = pd.DataFrame(all_results)
df.to_csv('results.csv')

print(df)
print(f"\nFDT - Average position of the first divergent token. The worst is 0.")
print(f"FDT norm - Average share of matched tokens until first divergent one. The best is 1.")
print(f"SDT -  Average number of divergent tokens in the evaluated outputs. The best is 0.")
print(f"SDTR norm - Average share of divergent tokens in the reference outputs. The best is 0, the maximum is 1.")

writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='all', index=False)
(max_row, max_col) = df.shape
wb = writer.book
worksheet = writer.sheets['all']
col_names = [{'header': col_name} for col_name in df.columns]
print(col_names)
# add table with coordinates: first row, first col, last row, last col;
#  header names or formatting can be inserted into dict
worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
    'columns': col_names,
    # 'style' = option Format as table value and is case sensitive
    # (look at the exact name into Excel)
    'style': None
})
worksheet.autofit()

wb.close()
