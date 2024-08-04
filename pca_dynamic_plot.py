import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys


def read_matrix_from_string(matrix_string):
    lines = matrix_string.strip().split("\n")
    headers = lines[0].strip().split()[1:]
    matrix = []
    for line in lines[1:]:
        matrix.append([float(x) if x != "" else 0 for x in line.strip().split()[1:]])
    return np.array(matrix), headers


def generate_pca_graph(matrix, headers, output_path):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(matrix)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)

    # Create a DataFrame for the principal components
    df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    df_pca["Label"] = headers[: len(matrix)]

    # Plot the PCA
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca["PC1"], df_pca["PC2"])

    # Annotate points with labels
    for i, label in enumerate(df_pca["Label"]):
        plt.annotate(label, (df_pca["PC1"][i], df_pca["PC2"][i]), fontsize=12)

    plt.title("PCA of Given Matrix")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # matrix_string = sys.argv[1]
    # output_path = sys.argv[2]
    matrix_string = """	Cht	A	B	C	D	E	F	G
                        A	0	0	0	0	0	0	0
                        B	80	0	0	0	0	0	0
                        C	80	20	0	0	0	0	0
                        D	60	60	100	0	0	0	0
                        E	40	40	80	0	0	0	0
                        F	0	20	0	60	60	0	0
                        G	100	40	20	0	40	40	0
                """
    output_path = ""

    # Read the matrix from the string input
    matrix, headers = read_matrix_from_string(matrix_string)

    # Generate the PCA graph
    generate_pca_graph(matrix, headers, output_path)


"""
Cht	A	B	C	D	E	F	G
A	0	0	0	0	0	0	0
B	80	0	0	0	0	0	0
C	80	20	0	0	0	0	0
D	60	60	100	0	0	0	0
E	40	40	80	0	0	0	0
F	0	20	0	60	60	0	0
G	100	40	20	0	40	40	0
"""
