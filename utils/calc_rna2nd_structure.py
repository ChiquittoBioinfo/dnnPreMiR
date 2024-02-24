import csv

# python -m pip install viennarna # ViennaRNA-2.6.3
import RNA ## RNAFold

import sys

# python calc_rna2nd_structure.py "../DataAugmentation/datasets/cnnMirtronPred/mirtrons.csv" "../human_pre_miRNA/CNN/models/pos.csv" True
# python calc_rna2nd_structure.py "../DataAugmentation/datasets/cnnMirtronPred/mirna_canonical.csv" "../human_pre_miRNA/CNN/models/neg.csv" False
# python calc_rna2nd_structure.py "../DataAugmentation/datasets/cnnMirtronPred/mirna_canonical.csv" "../human_pre_miRNA/CNN/models/neg.csv" False

def main(input_file, output_file, input_class):
    output = []
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seq = row['seq'].upper().replace('T', 'U')
            rnafold = RNA.fold(seq)

            seq_struc = [ "".join(s) for s in zip(seq, rnafold[0]) ]
            seq_struc = " ".join(seq_struc)

            output.append({
                'id': row['id'],
                # 'seq': seq,
                'seq_struc': seq_struc,
                'Classification': input_class
            })

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = output[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(output)

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    input_class = sys.argv[3].upper() == "TRUE"

    print(f"input_file={input_file}")
    print(f"output_file={output_file}")
    print(f"input_class={input_class}")

    main(input_file, output_file, input_class)
