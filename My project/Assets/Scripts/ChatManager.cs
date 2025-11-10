using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using Unity.Collections;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;

public class ChatManager : NetworkBehaviour
{
    [Header("UI References")]
    [SerializeField] private TMP_InputField messageInputField;
    [SerializeField] private Button sendButton;
    [SerializeField] private Transform messageContent;
    [SerializeField] private GameObject messagePrefab;
    [SerializeField] private ScrollRect scrollRect;

    [Header("Countdown UI")]
    [SerializeField] private GameObject countdownPanel;
    [SerializeField] private TMP_Text countdownText;
    [SerializeField] private float countdownTime = 45f;

    [Header("Settings")]
    [SerializeField] private int maxMessages = 50;
    [SerializeField] private bool useAIChat = true;

    [Header("Results")]
    [SerializeField] private GameObject resultsPanel;
    [SerializeField] private Button voteHumanButton;
    [SerializeField] private Button voteAIButton;
    [SerializeField] private TMP_Text resultText;

    private ChatClient client;
    private bool sendToServer = false;

    private NetworkVariable<int> totalVotes = new NetworkVariable<int>(0);
    private NetworkVariable<int> correctVotes = new NetworkVariable<int>(0);
    private HashSet<ulong> playersVoted = new HashSet<ulong>();

    private void Start()
    {
        StartCoroutine(StartCountdown());

        sendButton.onClick.AddListener(SendMessages);
        messageInputField.onSubmit.AddListener(delegate { SendMessages(); });
        messageInputField.ActivateInputField();

        UnityMainThreadDispatcher.Instance();

        SimpleNetworkManager netManager = FindObjectOfType<SimpleNetworkManager>();
        if (netManager != null)
        {
            if (netManager.impostorClientId.Value == ulong.MaxValue)
            {
                InitializeAIChat();
                sendToServer = true;
            }
            else
            {
                sendToServer = false;
            }
        }

        if (resultsPanel != null)
            resultsPanel.SetActive(false);

        if (voteHumanButton != null)
            voteHumanButton.onClick.AddListener(() => SubmitVote(false));

        if (voteAIButton != null)
            voteAIButton.onClick.AddListener(() => SubmitVote(true));
    }


    private void InitializeAIChat()
    {
        Debug.Log("=== Initializing AI Chat (TCP client) ===");
        client = new ChatClient();
        sendToServer = client.Connect("127.0.0.1", 65432);

        if (sendToServer)
        {
            Debug.Log("Successfully connected to AI server");
            client.OnMessageReceived += OnAIMessageReceived;
            Debug.Log("Event subscription complete");
        }
        else
        {
            Debug.LogWarning("Failed to connect to AI server");
        }
    }

    private void OnAIMessageReceived(string response)
    {
        Debug.Log($"=== OnAIMessageReceived called ===");
        Debug.Log($"Response: {response}");
        Debug.Log($"Thread ID: {System.Threading.Thread.CurrentThread.ManagedThreadId}");
        
        // Use the dispatcher to update UI on main thread
        try
        {
            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                Debug.Log($"=== Executing on main thread ===");
                Debug.Log($"Main Thread ID: {System.Threading.Thread.CurrentThread.ManagedThreadId}");
                Debug.Log($"About to call AddMessageToChat with: AI - {response}");
                AddMessageToChat("AI", response);
            });
            Debug.Log("Message enqueued to main thread");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error enqueueing message: {e.Message}\n{e.StackTrace}");
        }
    }

    private IEnumerator StartCountdown()
    {
        float remaining = countdownTime;

        if (countdownPanel != null)
        {
            countdownPanel.SetActive(true);
        }

        while (remaining > 0)
        {
            if (countdownText != null)
            {
                countdownText.text = Mathf.Ceil(remaining).ToString();
            }
            yield return new WaitForSeconds(1f);
            remaining--;
        }

        if (countdownPanel != null)
            countdownPanel.SetActive(false);
        

        if (resultsPanel != null)
            resultsPanel.SetActive(true);
    }

    public void StartVotingPhase()
    {
        if (resultText != null)
            resultText.text = "";
    }

    private void SubmitVote(bool votedAI)
    {
        if (resultsPanel != null)
            resultsPanel.SetActive(false);

        Debug.Log($"Player {NetworkManager.Singleton.LocalClientId} voted {(votedAI ? "IA" : "Humano")}");
        SubmitVoteServerRpc(votedAI, NetworkManager.Singleton.LocalClientId);
    }

    [ServerRpc(RequireOwnership = false)]
    private void SubmitVoteServerRpc(bool votedAI, ulong senderId)
    {
        if (playersVoted.Contains(senderId))
            return; // ya votó

        playersVoted.Add(senderId);
        totalVotes.Value++;

        // Revisar si el voto fue correcto
        SimpleNetworkManager netManager = FindObjectOfType<SimpleNetworkManager>();
        bool isAI = netManager != null && netManager.impostorClientId.Value == ulong.MaxValue;

        bool voteIsCorrect = (votedAI && isAI) || (!votedAI && !isAI);
        if (voteIsCorrect)
            correctVotes.Value++;

        // Si todos votaron, mostrar resultado
        if (totalVotes.Value >= NetworkManager.Singleton.ConnectedClients.Count)
        {
            bool allCorrect = correctVotes.Value == totalVotes.Value;
            ShowVotingResultClientRpc(allCorrect, correctVotes.Value, totalVotes.Value);

            // Reiniciar para futuras rondas si quisieras
            playersVoted.Clear();
            totalVotes.Value = 0;
            correctVotes.Value = 0;
        }
    }

    [ClientRpc]
    private void ShowVotingResultClientRpc(bool allCorrect, int correct, int total)
    {
        if (resultsPanel != null)
            resultsPanel.SetActive(true);

        string impostorType = "unknown";
        SimpleNetworkManager netManager = FindObjectOfType<SimpleNetworkManager>();
        if (netManager != null)
        {
            impostorType = (netManager.impostorClientId.Value == ulong.MaxValue) ? "IA" : "human";
        }

        if (resultText != null)
        {
            if (allCorrect)
                resultText.text = $"¡Everyone got it right! ({correct}/{total})\n\nThe impostor was: {impostorType}";
            else
                resultText.text = $"Some failed ({correct}/{total})\n\nThe impostor was: {impostorType}";
        }

        StartCoroutine(ReturnToMenuAfterDelay(10f));
    }

    private IEnumerator ReturnToMenuAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        SceneManager.LoadScene("ConnectionMenu");
    }

    public void SendMessages()
    {
        string messageText = messageInputField.text.Trim();
    
        if (string.IsNullOrEmpty(messageText))
            return;

        Debug.Log($"=== SendMessage called with: {messageText} ===");

        // Send to AI server if connected
        if (sendToServer && client != null)
        {
            Debug.Log($"Sending message to AI server: {messageText}");
            client.SendMessage(messageText);
        }
        else
        {
            Debug.Log($"Not sending to AI server (sendToServer: {sendToServer}, client: {client != null})");
        }

        // Send message to Netcode network (other players)
        if (NetworkManager.Singleton != null && NetworkManager.Singleton.IsConnectedClient)
        {
            Debug.Log("Sending via NetworkManager");
            SendMessageServerRpc(messageText);
        }
        else
        {
            // If not connected to network, just show locally
            Debug.Log("Showing message locally only");
            AddMessageToChat("You", messageText);
        }
    
        messageInputField.text = "";
        messageInputField.ActivateInputField();
    }

    [ServerRpc(RequireOwnership = false)]
    private void SendMessageServerRpc(string message, ServerRpcParams rpcParams = default)
    {
        // Get the client ID that sent the message
        ulong senderId = rpcParams.Receive.SenderClientId;
        
        Debug.Log($"Server received message from client {senderId}: {message}");
        
        // Broadcast to ALL clients (including sender)
        ReceiveMessageClientRpc(senderId, message);
    }

    [ClientRpc]
    private void ReceiveMessageClientRpc(ulong senderId, string message)
    {
        Debug.Log($"Client received message from {senderId}: {message}");
        
        // Determine sender name
        string senderName;
        if (NetworkManager.Singleton != null && senderId == NetworkManager.Singleton.LocalClientId)
        {
            senderName = "You";
        }
        else
        {
            senderName = $"User {senderId}";
        }
        
        AddMessageToChat(senderName, message);
    }

    private void AddMessageToChat(string sender, string message)
    {
        Debug.Log($"=== AddMessageToChat START ===");
        Debug.Log($"Sender: '{sender}'");
        Debug.Log($"Message: '{message}'");
        Debug.Log($"messageContent null? {messageContent == null}");
        Debug.Log($"messagePrefab null? {messagePrefab == null}");
        
        if (messageContent == null)
        {
            Debug.LogError("messageContent is NULL! Cannot add message.");
            return;
        }

        if (messagePrefab == null)
        {
            Debug.LogError("messagePrefab is NULL! Cannot add message.");
            return;
        }

        // Limit number of messages
        if (messageContent.childCount >= maxMessages)
        {
            Debug.Log($"Removing oldest message (current count: {messageContent.childCount})");
            Destroy(messageContent.GetChild(0).gameObject);
        }

        // Create new message
        Debug.Log("Instantiating message prefab...");
        GameObject newMessage = Instantiate(messagePrefab, messageContent);
        Debug.Log($"Message instantiated: {newMessage.name}");
        
        TMP_Text messageText = newMessage.GetComponentInChildren<TMP_Text>();
        
        if (messageText != null)
        {
            string formattedMessage = $"<b>{sender}:</b> {message}";
            messageText.text = formattedMessage;
            Debug.Log($"Message text set: {formattedMessage}");
            Debug.Log($"Message active in hierarchy? {newMessage.activeInHierarchy}");
            Debug.Log($"Message parent: {newMessage.transform.parent.name}");
            Debug.Log($"Total messages in chat: {messageContent.childCount}");
        }
        else
        {
            Debug.LogError("Could not find TMP_Text component in message prefab!");
            Debug.LogError($"Prefab structure: {GetGameObjectHierarchy(newMessage)}");
        }

        // Scroll to bottom
        if (scrollRect != null)
        {
            Canvas.ForceUpdateCanvases();
            scrollRect.verticalNormalizedPosition = 0f;
            Debug.Log("Scrolled to bottom");
        }
        
        Debug.Log($"=== AddMessageToChat END ===\n");
    }

    private string GetGameObjectHierarchy(GameObject obj, int level = 0)
    {
        string indent = new string(' ', level * 2);
        string result = $"{indent}- {obj.name}\n";
        
        foreach (Transform child in obj.transform)
        {
            result += GetGameObjectHierarchy(child.gameObject, level + 1);
        }
        
        return result;
    }

    private void Update()
    {
        // Send with Enter (without Shift)
        if (Input.GetKeyDown(KeyCode.Return) && 
            !Input.GetKey(KeyCode.LeftShift) && 
            !Input.GetKey(KeyCode.RightShift))
        {
            if (!messageInputField.isFocused)
            {
                messageInputField.ActivateInputField();
            }
        }
    }

    private void OnDestroy()
    {
        // Clean up TCP connection
        if (client != null)
        {
            Debug.Log("Cleaning up ChatClient");
            client.OnMessageReceived -= OnAIMessageReceived;
            client.Disconnect();
        }
    }
}