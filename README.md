<div align="center">
  <img src="https://github.com/user-attachments/assets/bc41024e-f07a-4bd0-bc89-f2f9f75e96d8" alt="image">
</div>




# 🚁 DroneVision - Secure Drone Segmentation for Rescue and Defence 🛡️

**Team Members:**
- Saikiran Mandula - LN73970
- Bharath Kumar Gopu - EY53339

## 📊 Research Question
How can drone-captured aerial images be used to segment and detect critical objects (such as people, vehicles, and infrastructure) to assist in faster and more efficient rescue and defense operations?

---

## 📖 Background
In response to the increasing demand for rapid and precise actions in **rescue** and **defense operations**, drones offer a promising solution for capturing real-time aerial imagery. The challenge, however, lies in accurately identifying and segmenting key objects from this data. This project aims to develop a system that processes drone images for **object detection** and **segmentation**, with a focus on:
- Locating people during emergencies
- Spotting potential threats in restricted areas

The functionality will be expanded by incorporating a web app for user interaction and potential **Power BI integration** for data visualization. 

---

## 🧠 Key Concepts
- **Semantic Segmentation**: Assigning labels to each pixel of an image for identifying objects and backgrounds.
- **Object Detection**: Detecting and classifying objects like people, vehicles, and infrastructure.
- **Rescue Operations**: Finding and saving people during disasters or emergencies.
- **Defense Surveillance**: Monitoring restricted areas for potential threats or unauthorized activity.

---

## 📂 Dataset Information
**Dataset Name**: [Semantic Drone Dataset](https://www.tugraz.at/index.php?id=22387) 

**Dataset Description**:
- **Image Resolution**: 6000x4000px (24Mpx)
- **Altitude Range**: 5 to 30 meters
- **Training Set**: 400 high-resolution images
- **Test Set**: 200 high-resolution images
- **Semantic Classes**: 20 classes (including people, cars, vegetation, buildings)
- **License**: Free for non-commercial research and academic use

This dataset is ideal for training and testing segmentation and object detection models, aligning perfectly with our goal of using drone imagery for rescue and defense operations.

---

## 🛠️ Methodology

### 1. **Data Preprocessing**
   - **Data Augmentation**: Techniques such as cropping, rotation, and scaling are used to increase the dataset’s diversity.
   - **Annotation Tool**: LabelImg used for object labeling and semantic segmentation annotation.

### 2. **Model Development**
   - **Base Model**: Implementing **U-Net** and **Mask R-CNN** for segmentation.
   - **Transfer Learning**: Fine-tuning pre-trained models like **DroneSegNet** to leverage existing knowledge and improve accuracy.
   - **Evaluation**: Metrics such as **IoU (Intersection over Union)** and **mAP (mean Average Precision)** to evaluate model performance.

### 3. **Web App Integration**
   - **Frontend**: Built with **React.js** for user interaction.
   - **Backend**: Flask API for processing images and returning segmented results.
   - **Visualization**: Integrating **Power BI** for real-time data analytics and insights.

---
## Project workflow:

![image](https://github.com/user-attachments/assets/157812b5-14fd-4a84-a428-511492de65fa)

## 💻 Tech Stack
- **Languages**: Python, JavaScript
- **Frameworks**: TensorFlow, Keras, React.js, Flask
- **Cloud Platforms**: AWS for hosting the drone image data
- **Libraries**: OpenCV, Scikit-Learn, Pandas
- **Visualization Tools**: Power BI, Matplotlib, Seaborn

---


## 🖼️ Example Images

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/7d9e34a6-4f03-48ac-84a6-f56a2a1503e6" alt="Drone-captured images" width="400">
      <br>
      <em>Figure 1: Drone-captured images</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/24003258-5f1c-48b7-8346-c5b19303423b" alt="Semantic segmentation of people" width="400">
      <br>
      <em>Figure 2: Semantic segmentation of people</em>
    </td>
  </tr>
</table>

---

## 🌐 Web Application Deployment

This app performs semantic segmentation on drone images using state-of-the-art deep learning models, identifying and classifying key areas within the scene.
<div align="center">
  <img src="https://github.com/user-attachments/assets/a8714364-0a03-4a07-a9b1-a640ea7dcdf7" alt="Web Application Interface" width="600">
  <p><em>Interactive Web Application for Drone Image Segmentation</em></p>
</div>

### How to Use:
1. **Select a Model**: Choose a segmentation model from the dropdown menu.  
2. **Upload an Image**: Drag and drop or upload a drone image.  
3. **View Results**: See the segmented output with color-coded classes.  

Try our web application: **[here](https://dronevisionsecure-drone-segmentation-for-rescue-and-defense-nx.streamlit.app/)**  

---

## 📚 Article Summaries

### DroneSegNet: Robust Aerial Semantic Segmentation for UAV-Based IoT Applications
- **Citation**: S. Chakravarthy, et al. (2022) in IEEE Transactions on Vehicular Technology, vol. 71, no. 4, pp. 4277-4286. [DOI](https://doi.org/10.1109/TVT.2022.3144358)
- **Summary**: This paper introduces **DroneSegNet**, a deep learning-based model for semantic segmentation of aerial images. It focuses on challenges such as varying scales and perspectives in complex environments. The robust segmentation methods from this paper could be adapted to enhance the accuracy of our system, improving safety and situational awareness in rescue and defense scenarios.

### Human Object Detection in Forest with Deep Learning Based on Drone’s Vision
- **Citation**: S. -P. Yong, et al. (2018) in 4th International Conference on Computer and Information Sciences (ICCOINS). [DOI](https://doi.org/10.1109/ICCOINS.2018.8510564)
- **Summary**: This research explores detecting humans in forest environments using deep learning and drone vision. Adapting their object detection techniques for urban or disaster environments will improve the accuracy and efficiency of our person detection module, crucial for rescue operations.

### Drone-surveillance for Search and Rescue in Natural Disasters
- **Citation**: Mishra et al. (2020) in Computer Communications, vol. 156. [DOI](https://doi.org/10.1016/j.comcom.2020.03.012)
- **Summary**: This paper highlights the role of drones in **search and rescue** during natural disasters, discussing how drones can provide real-time aerial imagery and data for decision-making. Integrating insights from this paper will help us improve the safety and effectiveness of our drone operations, especially in emergencies.


---

## 📝 Conclusions and Recommendations

### 1. Summary of Key Findings
The DroneVision project demonstrates the potential of semantic segmentation in UAV-based rescue and defense. **U-Net** achieved the highest pixel accuracy (83.3%), and a web application was developed for real-time result visualization and downloads, enhancing practical utility.

### 2. Recommendations
- **Optimize Models**: Use lightweight models for edge devices.  
- **Enhance Data**: Expand datasets to improve robustness.  
- **Integrate Features**: Combine segmentation with object detection for situational awareness.  

### 3. Future Work
- **3D Segmentation**: Explore 3D segmentation and depth estimation.  
- **Real-Time Pipelines**: Develop real-time pipelines with GPS integration.  
- **Thermal Imaging**: Incorporate thermal imaging for disaster response.  



