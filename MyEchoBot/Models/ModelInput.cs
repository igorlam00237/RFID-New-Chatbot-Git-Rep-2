using Microsoft.ML.Data;

namespace MonChatBot.Models
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }  // Assurez-vous également que la propriété Label est correctement décorée si elle est utilisée.
    }
}
