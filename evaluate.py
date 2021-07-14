import os
import argparse
import csv
import dtk.filesystem as dfs
from collections import Counter
import progressbar
import fae


def parse_annotation(annotations_file, excluded_annotations=["sil", "sp"]):
    phrase = []
    with open(annotations_file, 'r') as csvfile:
        annotation_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in annotation_reader:
            if row[2] in excluded_annotations:
                continue
            phrase.append(row[2])

    return " ".join(phrase)


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", help="Folder containing the video files")
parser.add_argument("--reference", "-r", nargs='?', help="(Optional) The folder containing reference video files")
parser.add_argument("--ext", "-e", nargs='+', help="(Optional) The video format extensions to search for")
parser.add_argument("--output", "-o", help="The directory to output the results to")
parser.add_argument("--lipreading", "-l", default="lrw", help='Model to use for lipreading')
parser.add_argument("--label_map", default="resources/500WordsSortedList.txt", help='Label mapping for words')
parser.add_argument("--filename_annotations", action='store_true', help='The annotations are in the filename')
parser.add_argument("--annotations", "-a", help='Folder containing annotations')
parser.add_argument("--annotation_ext", default=".align", help='Folder containing annotations')
parser.add_argument("--gpu", action='store_true', help='should the GPU be used (faster mouth stats)')
args = parser.parse_args()

folders = [args.input]
exts = [args.ext]
metrics = {}

if args.reference is not None:
    exts += [args.ext]
    folders += [args.reference]

if args.lipreading is not None:
    mouth_evaluator = fae.MouthEvaluator(args.lipreading)

final_metrics = metrics.copy()
files = dfs.list_matching_files(folders, ext=exts)

if args.output is not None:
    if os.path.dirname(args.output) != "" and not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    with open(args.output, 'w') as csvfile:
        line_writer = csv.writer(csvfile, delimiter=',')
        line_writer.writerow(metrics.keys())

bar = progressbar.ProgressBar(max_value=len(files["files"]))
for i in range(len(files["files"])):
    bar.update(i)
    video_path = os.path.join(folders[0], files["dirs"][i], files["files"][i]) + files["exts"][i][0]
    if args.reference is not None:
        ref_video_path = os.path.join(folders[1], files["dirs"][i], files["files"][i]) + files["exts"][i][1]
    else:
        ref_video_path = None

    if args.annotations is not None:
        annotation_path = os.path.join(folders[2], files["dirs"][i], files["files"][i]) + files["exts"][i][2]
        annotation = parse_annotation(annotation_path)
    elif args.filename_annotations:
        fname = os.path.splitext(os.path.basename(video_path))[0]
        annotation = ''.join([i for i in fname if not i.isdigit()]).replace("_", " ").rstrip()
    else:
        annotation = None

    if args.lipreading is not None:
        try:
            mouth_metrics = mouth_evaluator(video_path, ref_vid=ref_video_path, annotation=annotation)
        except Exception as e:
            print(e)

        metrics.update(mouth_metrics)

    if args.output is not None:
        with open(args.output, 'a') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=',')
            line_writer.writerow([metrics[k] for k in metrics.keys()])

    final_metrics = Counter(metrics) + Counter(final_metrics)

for k in final_metrics.keys():
    final_metrics[k] /= len(files["files"])

print("\n=================================RESULTS=================================")
print(','.join(map(str, final_metrics.keys())))
print(','.join(map(str, [final_metrics[x] for x in final_metrics.keys()])))
