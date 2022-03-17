# Reference: https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR#scrollTo=swiAP0RYqqVM
# Step 1: Build classifier

import os
import csv
from pose_embedding import FullBodyPoseEmbedder
from pose_classification import PoseClassifier
from bootstrap_helper import BootstrapHelper


def dump_for_the_app(pose_samples_folder, pose_samples_csv_path):
  file_extension = 'csv'
  file_separator = ','

  # Each file in the folder represents one pose class.
  file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

  with open(pose_samples_csv_path, 'w') as csv_out:
    csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
    for file_name in file_names:
      # Use file name as pose class name.
      class_name = file_name[:-(len(file_extension) + 1)]

      # One file line: `sample_00001,x1,y1,x2,y2,....`.
      with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
        csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
        for row in csv_in_reader:
          row.insert(1, class_name)
          csv_out_writer.writerow(row)
  print('Export csv done!')


def main(bootstrap_images_in_folder, bootstrap_images_out_folder, bootstrap_csvs_out_folder, export_csv):
  # Initialize helper.
  bootstrap_helper = BootstrapHelper(
      images_in_folder=bootstrap_images_in_folder,
      images_out_folder=bootstrap_images_out_folder,
      csvs_out_folder=bootstrap_csvs_out_folder,
  )

  # Check how many pose classes and images for them are available.
  print('\nCheck how many pose classes and images for them are available:')
  bootstrap_helper.print_images_in_statistics()
  
  # Bootstrap all images.
  # Set limit to some small number for debug.
  bootstrap_helper.bootstrap()

  # Check how many images were bootstrapped.
  print('\nCheck how many images were bootstrapped:')
  bootstrap_helper.print_images_out_statistics()

  # After initial bootstrapping images without detected poses were still saved
  # in the folder (but not in the CSVs) for debug purpose. Let's remove them.
  print('\nAlign CSVs with filtered images:')
  bootstrap_helper.align_images_and_csvs(print_removed_items=False)
  bootstrap_helper.print_images_out_statistics()
  
  # Automatic filtrationBootstrapHelper
  # Transforms pose landmarks into embedding.
  pose_embedder = FullBodyPoseEmbedder()

  # Classifies give pose against database of poses.
  pose_classifier = PoseClassifier(
      pose_samples_folder=bootstrap_csvs_out_folder,
      pose_embedder=pose_embedder,
      top_n_by_max_distance=30,
      top_n_by_mean_distance=10)

  outliers = pose_classifier.find_pose_sample_outliers()
  print('Number of outliers: ', len(outliers))

  # Analyze outliers.
  bootstrap_helper.analyze_outliers(outliers)

  # Remove all outliers (if you don't want to manually pick).
  bootstrap_helper.remove_outliers(outliers)

  # Align CSVs with images after removing outliers.
  bootstrap_helper.align_images_and_csvs(print_removed_items=False)
  bootstrap_helper.print_images_out_statistics()
  
  # Dump for the App
  dump_for_the_app(pose_samples_folder=bootstrap_csvs_out_folder, pose_samples_csv_path=export_csv)


if __name__ == '__main__':
  # Required structure of the images_in_folder:
  #
  #   extract_images/
  #     pushups_up/
  #       image_001.jpg
  #       image_002.jpg
  #       ...
  #     pushups_down/
  #       image_001.jpg
  #       image_002.jpg
  #       ...
  #     ...
  # Specific folder (e.g., extract_images) need have at least a pose class or many different pose class folder.
  bootstrap_images_in_folder = './resource/extract_images/'

  # Output folders for bootstrapped images and CSVs for log.
  bootstrap_images_out_folder = './log/knn_out_imgs_log'
  bootstrap_csvs_out_folder = './log/knn_out_csvs_log'

  # The finally output pose label csv file we want.
  export_csv = './result_pose.csv'

  main(bootstrap_images_in_folder, bootstrap_images_out_folder, bootstrap_csvs_out_folder, export_csv)
