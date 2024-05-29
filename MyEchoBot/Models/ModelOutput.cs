using Microsoft.ML.Data;

namespace MonChatBot.Models
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
    }
}
