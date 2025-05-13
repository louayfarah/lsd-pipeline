def validate_answers(df):
    """
    Adds an `is_validated` column which is 1 if all three columns
    agree: answerKey, llama_pred, and deepseek_pred; else 0.
    """
    df['is_validated'] = (
        (df['answerKey'] == df['llama_pred']) &
        (df['answerKey'] == df['deepseek_pred'])
    ).astype(int)

    df['is_validated_llama'] = (
        df['answerKey'] == df['llama_pred']
    ).astype(int)

    df['is_validated_deepseek'] = (
        df['answerKey'] == df['deepseek_pred']
    ).astype(int)
            
    return df


def combine_validated_en_fr(df_en, df_fr):
    pass