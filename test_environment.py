import sys
import torch

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        gpu_count = torch.cuda.device_count()
        if gpu_count==0:
            raise TypeError("Please add more GPUs to your system. Pytroch found: {} GPUs".format(gpu_count))
        else:
            print ("GPUs detected: {}".format(gpu_count))
            print(">>> Development environment passes all tests!")

torch.cuda.device_count()        

if __name__ == '__main__':
    main()