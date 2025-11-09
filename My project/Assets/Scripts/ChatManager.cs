using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using Unity.Collections;
using System.Collections;

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

    private ChatClient client;
    private bool sendToServer = false;

    private void Start()
    {
        StartCoroutine(StartCountdown());

        sendButton.onClick.AddListener(SendMessage);
        messageInputField.onSubmit.AddListener(delegate { SendMessage(); });
        messageInputField.ActivateInputField();

        UnityMainThreadDispatcher.Instance();
        Debug.Log("Main thread dispatcher initialized");

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
        {
            countdownPanel.SetActive(false);
        }
    }

    public void SendMessage()
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