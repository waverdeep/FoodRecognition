import logging


def setup_log(save_filename):
    # 로거 생성하기
    my_logger = logging.getLogger('waver')
    my_logger.setLevel(logging.INFO)

    # 파일 생성 핸들러 생성
    file_handler = logging.FileHandler(save_filename)
    console_handler = logging.StreamHandler()
    # 포맷 지정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    file_handler.setFormatter(formatter)

    # 핸들러 등록
    my_logger.addHandler(file_handler)
    my_logger.addHandler(console_handler)
    return my_logger


if __name__ == '__main__':
    format_logger = setup_log('./my.log')
    format_logger.info('hello world')
