from L1LLPJetTagger.core import add
from L1LLPJetTagger import delta_phi

if __name__ == "__main__":
    print("2 + 3 =", add(2, 3))
    print("5 + 4 =", add(5, 4))
    print("delta_phi(5,1) = ", delta_phi(5, 1))
    print("delta_phi(1,5) = ", delta_phi(1, 5))
