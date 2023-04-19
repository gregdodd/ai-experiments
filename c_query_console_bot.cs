using System;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace LM_Manual_Bot
{
    class Program
    {
        private static readonly string PineconeIndexName = "";
        private static readonly string EmbeddingModel = "";
        private static readonly string OpenAIApiKey = "";
        private static readonly string PineconeApiKey = "";
        private static readonly string PineconeProject = "";
        private static readonly string PineconeEnvironment = "";

        private static HttpClient httpClient = new HttpClient();

        static async Task Main(string[] args)
        {
            httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", OpenAIApiKey);
            httpClient.DefaultRequestHeaders.Add("Api-Key", PineconeApiKey);

            while (true)
            {
                Console.Write("Enter your query: ");
                string query = Console.ReadLine();

                if (query.ToLower() == "quit")
                {
                    break;
                }

                string answer = await Answer(query);
                Console.WriteLine($"Answer: {answer}\n");
            }
        }

        static async Task<string[]> RetrieveContext(string query)
        {
            string openaiUrl = "https://api.openai.com/v1/embeddings";

            var payload = new
            {
                input = query,
                model = EmbeddingModel
            };
            var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

            var response = await httpClient.PostAsync(openaiUrl, content);
            var jsonResponse = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<OpenAIEmbedResponse>(jsonResponse);
            float[] xq = result.data[0].embedding;

            string pineconeUrl = $"https://{PineconeIndexName}-{PineconeProject}.svc.{PineconeEnvironment}.pinecone.io/query";
            int topK = 3;
            bool includeValues = true;

            var pineconeResponse = await SendPineconeQuery(pineconeUrl, xq, topK, includeValues);
            var pineconeResult = JsonSerializer.Deserialize<PineconeQueryResponse>(pineconeResponse);

            var contexts = pineconeResult.matches.Select(x => x.metadata.GetProperty("text").ToString()).ToArray();
            return contexts;
        }

        static async Task<string> SendPineconeQuery(string url, float[] vector, int topK, bool includeValues)
        {
            var payload = new
            {
                vector = vector,
                topK = topK,
                includeValues = includeValues,
                includeMetadata = true
            };

            var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            var response = await httpClient.PostAsync(url, content);
            var jsonResponse = await response.Content.ReadAsStringAsync();

            return jsonResponse;
        }

        static async Task<string> Complete(string prompt)
        {
            string openaiUrl = "https://api.openai.com/v1/engines/text-davinci-003/completions";
            var payload = new
            {
                prompt = prompt,
                temperature = 0,
                max_tokens = 400,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0
            };
            var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

            httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", OpenAIApiKey);
            var response = await httpClient.PostAsync(openaiUrl, content);
            var jsonResponse = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<OpenAICompletionResponse>(jsonResponse);

            return result.choices[0].text.Trim();
        }

        static string BuildPrompt(string query, string[] context)
        {
            int contextLimit = 3750;
            string promptStart = "Answer the question using only the context provided.\n\n" +
                                 "Format instructions as an ordered list when possible.\n\n" +
                                 "If you do not know the answer, respond with: I'm sorry, I don't have an answer for that.\n\n" +
                                 "Context:\n";
            string promptEnd = $"\n\nQuestion: {query}\nAnswer:\n";

            string prompt = null;

            for (int i = 1; i < context.Length; i++)
            {
                if (string.Join("\n\n---\n\n", context.Take(i)).Length >= contextLimit)
                {
                    prompt = promptStart + string.Join("\n\n---\n\n", context.Take(i - 1)) + promptEnd;
                    break;
                }
                else if (i == context.Length - 1)
                {
                    prompt = promptStart + string.Join("\n\n---\n\n", context) + promptEnd;
                }
            }

            return prompt;
        }

        static async Task<string> Answer(string query)
        {
            string[] context = await RetrieveContext(query);
            string prompt = BuildPrompt(query, context);
            string answer = await Complete(prompt);

            return answer;
        }
    }

    // Helper classes for deserializing JSON responses
    public class OpenAIEmbedResponse
    {
        public OpenAIEmbedData[] data { get; set; }
    }

    public class OpenAIEmbedData
    {
        public float[] embedding { get; set; }
    }

    public class PineconeQueryResponse
    {
        public PineconeQueryMatch[] matches { get; set; }
    }

    public class PineconeQueryMatch
    {
        public float score { get; set; }
        public string id { get; set; }
        public JsonElement metadata { get; set; }
    }

    public class OpenAICompletionResponse
    {
        public OpenAICompletionChoice[] choices { get; set; }
    }

    public class OpenAICompletionChoice
    {
        public string text { get; set; }
    }
}