import os
import argparse
import csv
import dtk.filesystem as dfs
from collections import Counter
import progressbar
import fae


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Folder containing the video files")
    parser.add_argument("--reference", "-r", nargs='?', help="(Optional) The folder containing reference video files")
    parser.add_argument("--ext", nargs='+', help="(Optional) The video format extensions to search for")
    parser.add_argument("--output", "-o", help="The directory to output the results to")
    parser.add_argument("--lipreading", "-l", help='Model to use for lipreading')
    parser.add_argument("--emotion_net", "-e", default="emonet", help='Model to use for emotion recognition')
    parser.add_argument('--num_emotions', type=int, default=8, help='The number of emotions supported')
    parser.add_argument("--emotion_aggregation", default="voting", help='Method of aggregation to use for emotion recognition')
    parser.add_argument("--ignore_emotions", nargs='+', help='emotions to ignore')
    parser.add_argument("--annotation", "-a", help='annotations')
    parser.add_argument("--gpu", action='store_true', help='should the GPU be used (faster stats)')
    parser.add_argument("--full_report", help='path to output the full report')

    args = parser.parse_args()

    folders = [args.input]
    exts = [args.ext]
    metrics = {}

    if args.gpu:
        device = "cuda"

    if args.reference is not None:
        exts += [args.ext]
        folders += [args.reference]

    mouth_evaluator = fae.MouthEvaluator(args.lipreading, device=device)
    emotion_evaluator = fae.EmotionEvaluator(emotion_recognizer=args.emotion_net, num_emotions=args.num_emotions,
                                             ignore_emotions=args.ignore_emotions,
                                             aggregation=args.emotion_aggregation, device=device)

    final_metrics = metrics.copy()
    files = dfs.list_matching_files(folders, ext=exts)

    if args.output is not None:
        if os.path.dirname(args.output) != "" and not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))

        with open(args.output, 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=',')
            line_writer.writerow(metrics.keys())

    bar = progressbar.ProgressBar(max_value=len(files["files"]))

    annotator = None
    if args.annotation is not None:
        annotator = fae.Annotator(args.annotation)

    if args.full_report is not None:
        report_file = open(args.full_report, 'w')
        csv_writer = csv.DictWriter(report_file,
                                    ["cpbd", "mse", "ssim", "psnr", "LMD", "WER", "ccc_valence", "sagr_valence", "ccc_arousal", "sagr_arousal",
                                     "emotion_accuracy"],
                                    extrasaction='ignore')
        csv_writer.writeheader()
    else:
        csv_writer = None
        report_file = None

    for i in range(len(files["files"])):
        bar.update(i)
        video_path = os.path.join(folders[0], files["dirs"][i], files["files"][i]) + files["exts"][i][0]
        nr_metrics = fae.calculate_no_reference_metrics(video_path)

        metrics.update(nr_metrics)

        if args.reference is not None:
            ref_video_path = os.path.join(folders[1], files["dirs"][i], files["files"][i]) + files["exts"][i][1]
            fr_metrics = fae.calculate_full_reference_metrics(video_path, ref_video_path)
            metrics.update(fr_metrics)
        else:
            ref_video_path = None

        if annotator is None:
            annotation, emotion = None, None
        else:
            annotation, emotion = annotator(video_path)

        if args.lipreading is not None:
            try:
                mouth_metrics, raw_metrics = mouth_evaluator(video_path, ref_vid=ref_video_path, annotation=annotation)

                metrics.update(mouth_metrics)
            except Exception as e:
                print(e)

        emotion_metrics, raw_emotion_metrics = emotion_evaluator(video_path, ref_vid=ref_video_path, annotation=emotion)

        metrics.update(emotion_metrics)
        if csv_writer is not None:
            csv_writer.writerow(metrics)

        if args.output is not None:
            with open(args.output, 'a') as csvfile:
                line_writer = csv.writer(csvfile, delimiter=',')
                line_writer.writerow([metrics[k] for k in metrics.keys()])

        final_metrics = Counter(metrics) + Counter(final_metrics)

    if csv_writer is not None:
        report_file.close()

    for k in final_metrics.keys():
        final_metrics[k] /= len(files["files"])

    print("\n====================RESULTS====================")
    print("{:<15} {:<10}".format('Metric', 'Value'))
    for k, v in final_metrics.items():
        print("{:<15} {:<10}".format(k, v))


if __name__ == '__main__':
    evaluate()
