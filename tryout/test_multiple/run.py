from magnons.process import Process
import os

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    p = Process(dir_path)
    print(p.get_all_kwargs())