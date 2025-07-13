from model.randomforest import RandomForest
import pandas as pd

def model_predict(data, df, name):
    print("==== Chained Multi-Label Prediction ====")

    # Predict "Intent" from user input
    
    print("\n[1] Intent Prediction")

    # Create RandomForest model using TF-IDF features and intent labels
    intent_model = RandomForest(
        "Intent_RF",                           # model name
        data.get_features(),                   # TF-IDF features
        data.get_labels(specific_label='Intent')  # FIXED: correct parameter name
    )
    
    # Train model using training data
    intent_model.train(data)

    # Predict intent on test data
    intent_preds = intent_model.predict(data.X_test)

    # Get the test rows from the end of the original group
    df_test = df.iloc[-data.X_test.shape[0]:].copy()

    # Ensure required input column exists
    if 'combined_text' not in df_test.columns:
        raise KeyError("Missing required column: 'combined_text' in DataFrame.")

    # Save predicted intent into a new column
    df_test['pred_intent'] = intent_preds

    # Print evaluation report
    intent_model.print_results(data)

    
    # Predict "Tone" using combined text + predicted intent

    print("\n[2] Tone Prediction")

    # Fill missing values to avoid errors
    df_test['combined_text'] = df_test['combined_text'].fillna('')
    df_test['pred_intent'] = df_test['pred_intent'].fillna('')

    # Combine text and predicted intent
    df_test['tone_input'] = df_test['combined_text'] + ' ' + df_test['pred_intent']

    # Convert to vector (TF-IDF)
    tone_embeddings = data.vectorizer.transform(df_test['tone_input'])

    # Create new ProcessedData object for tone prediction
    tone_data = data.clone_with_new_features(tone_embeddings, new_label='Tone')

    # Train tone model and predict
    tone_model = RandomForest("Tone_RF", tone_embeddings, tone_data.get_labels(specific_label='Tone'))  
    tone_model.train(tone_data)
    tone_preds = tone_model.predict(tone_data.X_test)

    # Save predicted tone
    df_test.loc[df_test.index[-tone_data.X_test.shape[0]:], 'pred_tone'] = tone_preds

    # Show results
    tone_model.print_results(tone_data)

    
    # Predict "Resolution" using all previous inputs
    
    print("\n[3] Resolution Prediction")

    # Make sure tone values are filled
    df_test['pred_tone'] = df_test['pred_tone'].fillna('')

    # Combine everything
    df_test['res_input'] = (
        df_test['combined_text'] + ' ' +
        df_test['pred_intent'] + ' ' +
        df_test['pred_tone']
    )

    # Convert to TF-IDF features
    res_embeddings = data.vectorizer.transform(df_test['res_input'])

    # Wrap into ProcessedData
    res_data = data.clone_with_new_features(res_embeddings, new_label='Resolution')

    # Train and predict resolution
    res_model = RandomForest(
        "Resolution_RF",
        res_embeddings,
        res_data.get_labels(specific_label='Resolution')  
    )
    res_model.train(res_data)
    res_preds = res_model.predict(res_data.X_test)

    # Save resolution predictions
    df_test.loc[df_test.index[-res_data.X_test.shape[0]:], 'pred_resolution'] = res_preds

    # Show evaluation results
    res_model.print_results(res_data)

    
    # Save everything to output CSV

    df_combined = df.copy()
    df_combined.update(df_test)

    df_combined.to_csv(f'out_{name}_predictions.csv', index=False)
    print(f"\nPredictions saved to out_{name}_predictions.csv")


def model_evaluate(model, data):
    """
    Optional: Show evaluation report.
    """
    model.print_results(data)
