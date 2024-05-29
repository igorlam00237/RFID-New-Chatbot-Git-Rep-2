using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Bot.Builder;
using Microsoft.Bot.Schema;
using Microsoft.ML;
using MonChatBot.MachineLearning;
using MonChatBot.Models;
using System;

namespace EchoBot.Bots
{
    public class EchoBot : ActivityHandler
    {
        private readonly ITransformer _model;
        private readonly MLContext _mlContext;

        public EchoBot()
        {
            // Charger le modèle
            _mlContext = new MLContext();
            ModelBuilder modelBuilder = new ModelBuilder("dataset2_chatbot.csv");
            _model = modelBuilder.BuildAndTrainModel();
        }

        protected override async Task OnMessageActivityAsync(ITurnContext<IMessageActivity> turnContext, CancellationToken cancellationToken)
        {
            var userMessage = turnContext.Activity.Text.ToLower();
            string replyText;

            // Prédire la réponse en utilisant le modèle
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
            var prediction = predictionEngine.Predict(new ModelInput { Text = userMessage });
            replyText = prediction.Prediction;

            await turnContext.SendActivityAsync(MessageFactory.Text(replyText, replyText), cancellationToken);
        }

        protected override async Task OnMembersAddedAsync(IList<ChannelAccount> membersAdded, ITurnContext<IConversationUpdateActivity> turnContext, CancellationToken cancellationToken)
        {
            var welcomeText = "Hello and welcome!";
            foreach (var member in membersAdded)
            {
                if (member.Id != turnContext.Activity.Recipient.Id)
                {
                    await turnContext.SendActivityAsync(MessageFactory.Text(welcomeText, welcomeText), cancellationToken);
                }
            }
        }
    }
}
