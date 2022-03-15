import torch


def main():
    test_model = torch.jit.load('./traced_res152_model.pt', map_location='cpu')
    sample_data = torch.randn(2, 3, 512, 512)
    print(test_model(sample_data).size())


if __name__ == '__main__':
    main()