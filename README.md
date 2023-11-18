# DashboardSaludCMV

[![Demo](https://img.youtube.com/vi/CQKd_QVPBcQ/hqdefault.jpg)](https://www.youtube.com/embed/CQKd_QVPBcQ)

The "DashboardSaludCMV" serves as a comprehensive visualization tool designed to streamline the analysis of patient feedback for CESFAM (Centros de Salud Familiar). As a culmination of a multi-step data collection and classification process, this interactive dashboard provides users with a detailed, filterable overview of patient sentiments, health center performances, and demographic distributions.

### Features
- **Interactive Filters**: Users can refine their data view using a variety of filters, including keyword search, date ranges, age, gender, health center affiliation (CESFAM), visit frequency, satisfaction levels, recommendations, and sentiment labels.
  
- **Custom Visualizations**: The dashboard dynamically generates a suite of visualizations that include:
  - **Sentiment Distribution Pie Chart**: A breakdown of patient sentiments, categorized by positive, negative, and neutral feedback.
  - **Word Cloud**: Visual representation of the most common words found in patient comments, offering a quick insight into prevalent topics.
  - **Age Distribution Histogram**: A graphical representation of patient demographics segmented by age.
  - **Satisfaction and Recommendation Ratings**: Bar charts that display aggregate satisfaction and recommendation scores across different CESFAMs.
  - **Visit Frequency and Gender Distribution Charts**: Insights into the frequency of visits and the gender distribution of patients providing feedback.
  - **Health Center Specific Feedback Analysis**: A bar chart visualizing the volume of feedback per health center.
  - **Average Satisfaction and Recommendation Scores**: Compares average scores across different CESFAMs, enabling benchmarking and performance tracking.
  - **Feedback Chronology Line Graph**: A temporal view of feedback frequency, highlighting trends over time.
  - **3D Scatter Plot**: An advanced visualization mapping age against satisfaction and recommendation scores, offering a multi-dimensional perspective on the data.
  
- **Data Table**: Beyond visualizations, the dashboard also displays a detailed data table with the ability to hide or show columns based on user preference.

### User Interface
The application boasts a clean and intuitive interface, crafted with Streamlit, ensuring that data exploration remains user-friendly and accessible to non-technical stakeholders. A sidebar contains all the filter controls, making it straightforward to drill down into specific data segments.

### Data Processing
The backend of the dashboard interfaces with a MongoDB database, retrieving and processing patient feedback data in real-time. Data transformation and cleaning are handled seamlessly within the application, ensuring the visuals are always based on the latest and most accurate data.

### Deployment
The dashboard is deployable as a web application, allowing for easy access by health administrators and staff within the CESFAM network. Its responsive design ensures compatibility across devices, from desktops to tablets, facilitating on-the-go analysis.
