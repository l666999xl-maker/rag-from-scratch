from config import BUILD_DIR, PDF_DIR, check


def main() -> None:
    print("=" * 50)
    print("  环境自检")
    print("=" * 50)

    missing = check()
    if missing:
        print("缺少配置：", "、".join(missing))
        print("请把 project7-test-rag/.env 复制到本项目根目录，改名为 .env")
        return

    print("密钥配置：OK")
    print("PDF 目录：", PDF_DIR)
    print("PDF 数量：", len(list(PDF_DIR.glob("*.pdf"))))
    print("中间产物目录：", BUILD_DIR)

    BUILD_DIR.mkdir(exist_ok=True)
    print("\n环境就绪，可以开始第 2 步。")


if __name__ == "__main__":
    main()