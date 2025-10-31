using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ChatManager : MonoBehaviour
{
    [Header("UI References")]
    [SerializeField] private TMP_InputField messageInputField;
    [SerializeField] private Button sendButton;
    [SerializeField] private Transform messageContent; // The Content object inside ScrollView
    [SerializeField] private GameObject messagePrefab;
    [SerializeField] private ScrollRect scrollRect;

    [Header("Settings")]
    [SerializeField] private int maxMessages = 50; // Limit messages to prevent performance issues

    private void Start()
    {
        // Add listener to send button
        sendButton.onClick.AddListener(SendMessage);
        
        // Allow sending with Enter key
        messageInputField.onSubmit.AddListener(delegate { SendMessage(); });
        
        // Focus input field at start
        messageInputField.ActivateInputField();
    }

    public void SendMessage()
    {
        string messageText = messageInputField.text.Trim();
        
        // Don't send empty messages
        if (string.IsNullOrEmpty(messageText))
        {
            return;
        }

        // Create new message
        AddMessageToChat("You", messageText);
        
        // Clear input field
        messageInputField.text = "";
        
        // Refocus input field
        messageInputField.ActivateInputField();
    }

    private void AddMessageToChat(string sender, string message)
    {
        // Limit number of messages
        if (messageContent.childCount >= maxMessages)
        {
            Destroy(messageContent.GetChild(0).gameObject);
        }

        // Instantiate message prefab
        GameObject newMessage = Instantiate(messagePrefab, messageContent);
        
        // Set message text
        TMP_Text messageText = newMessage.GetComponentInChildren<TMP_Text>();
        if (messageText != null)
        {
            messageText.text = $"<b>{sender}:</b> {message}";
        }

        // Scroll to bottom
        Canvas.ForceUpdateCanvases();
        scrollRect.verticalNormalizedPosition = 0f;
    }

    private void Update()
    {
        // Alternative: Send with Enter, new line with Shift+Enter
        if (Input.GetKeyDown(KeyCode.Return) && !Input.GetKey(KeyCode.LeftShift) && !Input.GetKey(KeyCode.RightShift))
        {
            if (!messageInputField.isFocused)
            {
                messageInputField.ActivateInputField();
            }
        }
    }
}