# Prediction and Results Section
if predict_btn:
    
    # 1. Map categorical text inputs to the numerical values used during training
    # (Adjust these mappings to match your exact training label encoder values)
    gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
    internet_mapping = {"No": 0, "Yes": 1}
    sleep_quality_mapping = {"Poor": 0, "Average": 1, "Good": 2}
    study_method_mapping = {"Self-study": 0, "Group study": 1, "Online videos": 2, "Textbooks": 3}
    exam_difficulty_mapping = {"Easy": 0, "Moderate": 1, "Hard": 2}
    
    # Optional: If 'course' was also label encoded, map it similarly. 
    # If it was text/one-hot encoded through a pipeline, leave it as a string.
    
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender_mapping[gender]],
        'course': [course], # Keep string if using a pipeline, or map to int if label encoded
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_mapping[internet_access]],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality_mapping[sleep_quality]],
        'study_method': [study_method_mapping[study_method]],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty_mapping[exam_difficulty]]
    })

    try:
        # Generate prediction
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")
        st.success(f"### Estimated Performance Score: **{prediction:.2f}**")
        
    except Exception as e:
        st.error(f"Prediction failed details: {e}")
