import streamlit as st
import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import tempfile
import subprocess
import shutil
import numpy as np
import pandas as pd

# -----------------------------
# GraphSAGE Model (SAME AS TRAINING)
# -----------------------------
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return self.classifier(x)

# -----------------------------
# Feature Extraction
# -----------------------------
def count_loc(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return len(f.readlines())
    except:
        return 0

def count_functions_classes(file_path):
    functions = 0
    classes = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
    except:
        pass

    return functions, classes

def extract_imports(file_path):
    imports = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except:
        pass

    return imports

# -----------------------------
# Build Graph from Folder
# -----------------------------
def build_graph_from_folder(folder_path):

    py_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    features = []
    edges = []
    file_index = {file: idx for idx, file in enumerate(py_files)}

    # First pass: extract raw metrics
    raw_data = {}

    for file in py_files:
        loc = count_loc(file)
        functions, classes = count_functions_classes(file)
        imports = extract_imports(file)

        raw_data[file] = {
            "loc": loc,
            "functions": functions,
            "classes": classes,
            "imports": imports
        }

    # Build edges based on simple import matching
    for file in py_files:
        idx = file_index[file]
        imports = raw_data[file]["imports"]

        for imp in imports:
            for target in py_files:
                if imp in target:
                    edges.append([idx, file_index[target]])

    # Compute degrees
    indegree = {file: 0 for file in py_files}
    outdegree = {file: 0 for file in py_files}

    for src, dst in edges:
        src_file = py_files[src]
        dst_file = py_files[dst]
        outdegree[src_file] += 1
        indegree[dst_file] += 1

    # Final 10-dimensional feature vector (MATCH TRAINING)
    for file in py_files:
        loc = raw_data[file]["loc"]
        functions = raw_data[file]["functions"]
        classes = raw_data[file]["classes"]
        imports_count = len(raw_data[file]["imports"])
        indeg = indegree[file]
        outdeg = outdegree[file]

        features.append([
            loc,
            functions,
            classes,
            imports_count,
            indeg,
            outdeg,
            loc / 1000.0,
            functions / (loc + 1),
            classes / (loc + 1),
            imports_count / (loc + 1)
        ])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index), py_files

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    model = GraphSAGE(input_dim=10, hidden_dim=64)  # MUST MATCH TRAINING
    model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    model.eval()
    return model

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Software Defect Risk Analyzer")
st.write("Upload a file, folder, or GitHub repository to analyze defect risk.")

model = load_model()

option = st.radio(
    "Select Input Type:",
    ("Upload Python File", "Upload Folder (ZIP)", "GitHub URL")
)

data = None
files = None

if option == "Upload Python File":
    uploaded_file = st.file_uploader("Upload .py file", type=["py"])

    if uploaded_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            data, files = build_graph_from_folder(tmpdir)

elif option == "Upload Folder (ZIP)":
    uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])

    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "repo.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            shutil.unpack_archive(zip_path, tmpdir)
            data, files = build_graph_from_folder(tmpdir)

elif option == "GitHub URL":
    github_url = st.text_input("Enter GitHub Repository URL")

    if github_url:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "clone", github_url, tmpdir], capture_output=True)
            data, files = build_graph_from_folder(tmpdir)

# -----------------------------
# Prediction
# -----------------------------
if data:
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1]
        avg_risk = probs.mean().item()

    st.subheader("📊 Project Risk Analysis")

    st.metric("Average Defect Risk Score", round(avg_risk, 4))

    if avg_risk < 0.3:
        st.success("Low Risk Project")
    elif avg_risk < 0.6:
        st.warning("Medium Risk Project")
    else:
        st.error("High Risk Project")

    # File-wise Risk Table
    st.subheader("📁 File-Level Risk Ranking")

    df = pd.DataFrame({
        "File": [os.path.basename(f) for f in files],
        "Risk Score": probs.numpy()
    })

    df = df.sort_values(by="Risk Score", ascending=False)
    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index("File"))
