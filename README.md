import os
import argparse

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB -> 5 * 1024 * 1024 字节

def split_file(input_path, output_dir=None, chunk_size=CHUNK_SIZE):
    # 检查源文件
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 默认输出目录：和源文件同目录下，建一个子目录
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(input_path))
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(base_dir, f"{base_name}_parts")

    os.makedirs(output_dir, exist_ok=True)

    part_index = 1
    with open(input_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            output_path = os.path.join(output_dir, f"{part_index}.bin")
            with open(output_path, "wb") as out_f:
                out_f.write(chunk)

            print(f"写出分片: {output_path}  大小: {len(chunk)} 字节")
            part_index += 1

    print(f"完成！共生成 {part_index - 1} 个分片，目录：{output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 .npz 文件按 5MB 拆分为多个 1.bin, 2.bin ... 文件"
    )
    parser.add_argument("input_path", help="需要拆分的 .npz 文件路径")
    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="输出目录（默认：在源文件同目录下创建 {文件名}_parts 文件夹）"
    )
    parser.add_argument(
        "-s", "--size_mb",
        type=float,
        default=5.0,
        help="每个分片大小（单位 MB，默认 5MB）"
    )

    args = parser.parse_args()
    chunk_bytes = int(args.size_mb * 1024 * 1024)

    split_file(args.input_path, args.output_dir, chunk_bytes)
