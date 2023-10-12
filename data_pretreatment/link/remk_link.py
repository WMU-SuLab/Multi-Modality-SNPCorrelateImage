import os
from datetime import datetime

import click


@click.command()
@click.argument('divide_dir_path')
@click.option('-g', '--new_gene_dir_path', default=None, type=str, help='new gene dir path to make symbol link')
@click.option('-i', '--new_image_dir_path', default=None, type=str, help='new image dir path to make symbol link')
def main(divide_dir_path: str, new_gene_dir_path: str, new_image_dir_path: str):
    """
    根据现有的软连接重新构建软连接
    :param divide_dir_path:具体的数据集地址，非divide大文件夹
    :param new_gene_dir_path:
    :param new_image_dir_path:
    :return:
    """
    divide_dir_path = os.path.realpath(divide_dir_path)
    dataset_dir_paths = [dataset_dir_path for dir_name in os.listdir(divide_dir_path)
                         if os.path.isdir(dataset_dir_path := os.path.join(divide_dir_path, dir_name))]
    for dataset_dir_path in dataset_dir_paths:
        for dir_name in os.listdir(dataset_dir_path):
            for file_name in os.listdir(os.path.join(dataset_dir_path, dir_name)):
                link_file_path = os.path.join(dataset_dir_path, dir_name, file_name)
                os.remove(link_file_path)
                if dir_name == 'gene' and new_gene_dir_path:
                    os.symlink(os.path.join(new_gene_dir_path, file_name), link_file_path)
                elif dir_name == 'image' and new_image_dir_path:
                    os.symlink(os.path.join(new_image_dir_path, file_name), link_file_path)
    with open('info.txt', 'a') as f:
        f.write(f"remake symlinks at {datetime.now().strftime('%Y%m%d%H%M%S')}\n")
        f.write(divide_dir_path + '\n')
        f.write(new_gene_dir_path + '\n')
        f.write(new_image_dir_path + '\n')


if __name__ == '__main__':
    main()
