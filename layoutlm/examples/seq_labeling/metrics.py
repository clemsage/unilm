import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def aggregate_few_shot_experiments(args):
    metric_values = {}
    metric_values["SROIE"] = {  # For following seeds: [42, 43, 44, 45, 46]
        "Pre-trained": {
            "recall": np.array(
                [
                    [0.6347, 0.7183, 0.8321, 0.8595, 0.8927, 0.9171, 0.9431],
                    [0.5850, 0.7003, 0.8148, 0.9099, 0.9056, 0.9035, 0.9388],
                    [0.6513, 0.7846, 0.7867, 0.8854, 0.9063, 0.9128, 0.9366],
                    [0.6110, 0.7313, 0.8084, 0.8638, 0.8905, 0.9186, 0.9222],
                    [0.5483, 0.7334, 0.8242, 0.8883, 0.9179, 0.9287, 0.9416],
                ]
            ),
            "precision": np.array(
                [
                    [0.7762, 0.8274, 0.8724, 0.8997, 0.9198, 0.9423, 0.9541],
                    [0.7733, 0.8209, 0.8673, 0.9219, 0.9318, 0.9414, 0.9518],
                    [0.8271, 0.8416, 0.8764, 0.9090, 0.9230, 0.9323, 0.9468],
                    [0.7485, 0.8423, 0.8772, 0.9070, 0.9028, 0.9327, 0.9343],
                    [0.7542, 0.8276, 0.8889, 0.9154, 0.9333, 0.9436, 0.9485],
                ]
            ),
            "f1_score": np.array(
                [
                    [0.6984, 0.7690, 0.8518, 0.8791, 0.9060, 0.9295, 0.9486],
                    [0.6661, 0.7558, 0.8403, 0.9159, 0.9185, 0.9221, 0.9452],
                    [0.7287, 0.8121, 0.8292, 0.8971, 0.9146, 0.9225, 0.9417],
                    [0.6727, 0.7829, 0.8414, 0.8849, 0.8966, 0.9256, 0.9282],
                    [0.6350, 0.7777, 0.8553, 0.9016, 0.9255, 0.9361, 0.9450],
                ]
            ),
        },
        "From scratch": {
            "recall": np.array(
                [
                    [0.0058, 0.0310, 0.0670, 0.1585, 0.2601, 0.3465, 0.4669],
                    [0.0029, 0.0043, 0.0915, 0.1542, 0.2543, 0.3422, 0.5108],
                    [0.0014, 0.0166, 0.0562, 0.1434, 0.2450, 0.3840, 0.4813],
                    [0.0007, 0.0101, 0.0620, 0.1268, 0.2615, 0.3732, 0.5180],
                    [0.0029, 0.0130, 0.0670, 0.1268, 0.2709, 0.3919, 0.5058],
                ]
            ),
            "precision": np.array(
                [
                    [0.1039, 0.1340, 0.1606, 0.2757, 0.4507, 0.5068, 0.6304],
                    [0.1026, 0.0531, 0.1308, 0.3092, 0.4153, 0.5331, 0.6370],
                    [0.0526, 0.0909, 0.1722, 0.3119, 0.4387, 0.5484, 0.6693],
                    [0.0110, 0.0690, 0.1803, 0.2378, 0.4206, 0.5756, 0.6590],
                    [0.1143, 0.1065, 0.1751, 0.2472, 0.4455, 0.5591, 0.6561],
                ]
            ),
            "f1_score": np.array(
                [
                    [0.0109, 0.0503, 0.0946, 0.2013, 0.3298, 0.4116, 0.5364],
                    [0.0056, 0.0080, 0.1077, 0.2058, 0.3155, 0.4168, 0.5670],
                    [0.0027, 0.0280, 0.0847, 0.1964, 0.3144, 0.4517, 0.5599],
                    [0.0013, 0.0176, 0.0922, 0.1654, 0.3225, 0.4528, 0.5801],
                    [0.0056, 0.0231, 0.0969, 0.1676, 0.3369, 0.4608, 0.5712],
                ]
            ),
        },
        "BLSTM": {
            "recall": np.array(
                [
                    [0.2860, 0.4244, 0.5908, 0.6650, 0.7529, 0.8300, 0.8422],
                    [0.3019, 0.3631, 0.5771, 0.7017, 0.7579, 0.7903, 0.8703],
                    [0.3516, 0.4330, 0.5944, 0.6981, 0.7176, 0.8278, 0.8660],
                    [0.3977, 0.4863, 0.6146, 0.6643, 0.7298, 0.8163, 0.8617],
                    [0.2673, 0.4409, 0.5857, 0.6988, 0.7615, 0.8004, 0.8552],
                ]
            ),
            "precision": np.array(
                [
                    [0.5272, 0.6320, 0.7656, 0.8175, 0.8723, 0.9007, 0.9197],
                    [0.5779, 0.6238, 0.8058, 0.8361, 0.8623, 0.8783, 0.9186],
                    [0.6272, 0.7013, 0.7346, 0.8205, 0.8426, 0.8977, 0.9211],
                    [0.7169, 0.7105, 0.7664, 0.8451, 0.8379, 0.8831, 0.9095],
                    [0.5752, 0.6777, 0.7735, 0.8524, 0.8580, 0.8721, 0.9194],
                ]
            ),
            "f1_score": np.array(
                [
                    [0.3709, 0.5078, 0.6669, 0.7334, 0.8082, 0.8639, 0.8793],
                    [0.3966, 0.4590, 0.6725, 0.7630, 0.8067, 0.8320, 0.8938],
                    [0.4506, 0.5354, 0.6571, 0.7544, 0.7751, 0.8613, 0.8927],
                    [0.5116, 0.5774, 0.6821, 0.7438, 0.7801, 0.8484, 0.8849],
                    [0.3650, 0.5343, 0.6667, 0.7680, 0.8069, 0.8347, 0.8862],
                ]
            ),
        },
    }

    models = ["Pre-trained", "From scratch", "BLSTM"]
    nbs_docs = {}
    nbs_docs["SROIE"] = [8, 16, 32, 64, 128, 256, 600]

    sns.set()  # Set seaborn theme
    for i, ds in enumerate(metric_values):
        means, stds = {}, {}
        plt.figure(i)
        for model in models:
            # check that we have not made data entry errors by recomputing f1 scores from reported recalls and
            # precisions
            recompute_f1_score = (
                    2
                    * metric_values[ds][model]["recall"]
                    * metric_values[ds][model]["precision"]
                    / (metric_values[ds][model]["recall"] + metric_values[ds][model]["precision"])
            )
            diffs = np.abs(recompute_f1_score - metric_values[ds][model]["f1_score"])
            assert np.max(diffs) < 1e-4

            means[model] = np.mean(metric_values[ds][model]["f1_score"], axis=0)
            stds[model] = np.std(metric_values[ds][model]["f1_score"], axis=0)

            plt.semilogx(nbs_docs[ds], means[model], label=model)
            plt.fill_between(nbs_docs[ds], means[model] - stds[model], means[model] + stds[model], alpha=0.2)

        plt.xticks(nbs_docs[ds], nbs_docs[ds])
        plt.xlabel("# training documents")
        plt.yticks(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        plt.ylabel("F1-score")
        plt.ylim(0, 1.05)
        plt.legend(loc=4)
        if False:
            plt.show()
        else:
            os.makedirs(args.plots_path, exist_ok=True)
            plt.savefig(os.path.join(args.plots_path, "few_shot_%s.png" % ds), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--plots_path", type=str, default="output/plots", help="Path to the output plots")

    args = parser.parse_args()
    aggregate_few_shot_experiments(args)
