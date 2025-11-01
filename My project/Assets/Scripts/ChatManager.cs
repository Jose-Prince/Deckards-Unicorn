using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using Unity.Collections;

public class ChatManager : NetworkBehaviour
{
    [Header("UI References")]
    [SerializeField] private TMP_InputField messageInputField;
    [SerializeField] private Button sendButton;
    [SerializeField] private Transform messageContent;
    [SerializeField] private GameObject messagePrefab;
    [SerializeField] private ScrollRect scrollRect;

    [Header("Settings")]
    [SerializeField] private int maxMessages = 50;

    private void Start()
    {
        sendButton.onClick.AddListener(SendMessage);
        messageInputField.onSubmit.AddListener(delegate { SendMessage(); });
        messageInputField.ActivateInputField();
    }

    public void SendMessage()
    {
        string messageText = messageInputField.text.Trim();
        
        if (string.IsNullOrEmpty(messageText))
            return;

        // Enviar mensaje al servidor
        if (NetworkManager.Singleton != null && NetworkManager.Singleton.IsConnectedClient)
        {
            SendMessageServerRpc(messageText);
        }
        else
        {
            // Si no hay conexión, solo muestra localmente (modo prueba)
            AddMessageToChat("You", messageText);
        }
        
        messageInputField.text = "";
        messageInputField.ActivateInputField();
    }

    [ServerRpc(RequireOwnership = false)]
    private void SendMessageServerRpc(string message, ServerRpcParams rpcParams = default)
    {
        // Obtener el ID del cliente que envió el mensaje
        ulong senderId = rpcParams.Receive.SenderClientId;
        
        Debug.Log($"Server received message from client {senderId}: {message}");
        
        // Reenviar a TODOS los clientes (incluyendo el que lo envió)
        ReceiveMessageClientRpc(senderId, message);
    }

    [ClientRpc]
    private void ReceiveMessageClientRpc(ulong senderId, string message)
    {
        Debug.Log($"Client received message from {senderId}: {message}");
        
        // Determinar el nombre del remitente
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
        // Limitar número de mensajes
        if (messageContent.childCount >= maxMessages)
        {
            Destroy(messageContent.GetChild(0).gameObject);
        }

        // Crear nuevo mensaje
        GameObject newMessage = Instantiate(messagePrefab, messageContent);
        TMP_Text messageText = newMessage.GetComponentInChildren<TMP_Text>();
        
        if (messageText != null)
        {
            messageText.text = $"<b>{sender}:</b> {message}";
        }

        // Scroll al final
        Canvas.ForceUpdateCanvases();
        scrollRect.verticalNormalizedPosition = 0f;
    }

    private void Update()
    {
        // Enviar con Enter (sin Shift)
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
}