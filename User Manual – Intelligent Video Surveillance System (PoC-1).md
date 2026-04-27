# **User Manual – Intelligent Video Surveillance System (PoC-1)**

---

## 1. Introduction

This system processes video input and performs object detection, tracking, and analytics using a Streamlit-based dashboard.
The project runs entirely in **Google Colab** and stores outputs in Google Drive.

---

## 2. Prerequisites

* Google account
* Google Colab access
* Internet connection

---

## 3. Setup

### Step 1: Open Notebook

Open the PoC-1 notebook in Google Colab.

---

### Step 2: Run Setup.py

This cell performs:

* Google Drive mounting
* Project folder creation
* Dependency installation

After execution, folders will be created:

* videos/
* output/

---

## 4. Execution (IMPORTANT – FOLLOW ORDER)

---

### Step 1: Run  Pipeline.py

* Creates `pipeline.py`
* Defines `run_pipeline()`
* Handles:

  * Detection (YOLOv8)
  * Tracking (ByteTrack)
  * Metrics calculation
  * Output generation

---

### Step 2: Run Install UI Dependencies.py

Installs:

* Streamlit
* Pandas
* Matplotlib

---

### Step 3: Run Dashboard Setup.py

* Creates `dashboard.py`
* Builds Streamlit UI
* Connects UI with pipeline

---

### Step 4: Run Launch Dashboard.py

This step:

* Starts Streamlit server
* Creates public URL using Cloudflare

👉 After execution, you will get a link like:

```
https://xxxx.trycloudflare.com
```

Open this link in browser.

---

## 5. Using the Dashboard

Inside the UI:

### Option 1: Upload Video

* Upload file (max 200MB)
* Click **Run Uploaded Video**

### Option 2: Use Sample Video

* Select from dropdown
* Click **Run Sample Video**

---

## 6. Outputs

The dashboard shows:

* Annotated video (tracking + IDs)
* Heatmap video
* Overlay video
* Heatmap image
* CSV preview
* Summary metrics

---

## 7. Output Storage

All outputs are saved in:

/content/drive/MyDrive/cv_project/output/run_<timestamp>/

Each run contains:

* Videos
* CSV
* Heatmap image

---

## 8. Important Notes

* Always run cells in order (Setup → Launch Dashboard.py)
* Do not skip any step
* Use GPU runtime for better performance
* Large videos take more time

---

## 9. Troubleshooting

* If URL not generated → rerun Launch Dashboard.py
* If Streamlit fails → restart runtime
* If no output → ensure pipeline ran correctly

---

## 10. Conclusion

The system provides a complete pipeline from video input to analytics output using an interactive UI.
By following the steps above, users can run the system without modifying the code.
