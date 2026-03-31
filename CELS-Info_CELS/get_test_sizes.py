import sys
from nte.experiment.utils import dataset_mapper


# datasets = [
#     "Coffee",
#     "GunPoint",
#     "ECG200",
#     "TwoLeadECG",
#     "CBF",
#     "Plane",
#     "BirdChicken",
#     "Lightning2",
#     "Chinatown",
#     "BeetleFly",
#     "FaceFour",
#     "Beef",
#     "Car",
#     "ArrowHead",
#     "Lightning7",
#     "Computers",
#     "OSULeaf",
#     "Worms",
#     "SwedishLeaf",
#     "Trace"
# ]

datasets = [
            # 'GunPointAgeSpan',
            # 'GunPointMaleVersusFemale',
            # 'GunPointOldVersusYoung',
            # 'HandOutlines',
            # 'Chinatown',
            # 'FordB',
            # 'FordA',
            # 'FreezerRegularTrain',
            'Wafer',
            ]

for ds in datasets:
    dataset = dataset_mapper(DATASET=ds)
    dataset.load_data()
    print(f"{ds} {len(dataset.test_data)}")
