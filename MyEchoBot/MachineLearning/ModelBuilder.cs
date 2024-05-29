using System;
using Microsoft.ML;
using MonChatBot.Models;

namespace MonChatBot.MachineLearning
{
    public class ModelBuilder
    {
        private readonly string _dataPath;
        private readonly MLContext _mlContext;

        public ModelBuilder(string dataPath)
        {
            _dataPath = dataPath;
            _mlContext = new MLContext(seed: 0);
        }

        public ITransformer BuildAndTrainModel()
        {
            // Charger les données
            IDataView dataView = _mlContext.Data.LoadFromTextFile<ModelInput>(
                _dataPath, hasHeader: true, separatorChar: ';');

            // Configurer le pipeline de traitement des données
            var dataProcessPipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelInput.Label))
                .Append(_mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.Text)))
                .AppendCacheCheckpoint(_mlContext);

            // Choisir un algorithme de machine learning et ajouter au pipeline
            var trainer = _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Entraîner le modèle
            ITransformer model = trainingPipeline.Fit(dataView);
            return model;
        }
    }
}
