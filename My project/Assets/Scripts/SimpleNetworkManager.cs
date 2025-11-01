using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;
using UnityEngine.SceneManagement;
using System.Collections.Generic;

public class SimpleNetworkManager : MonoBehaviour
{
    [Header("UI References - Connection")]
    [SerializeField] private Button hostButton;
    [SerializeField] private Button joinButton;
    [SerializeField] private TMP_InputField ipAddressInput;
    [SerializeField] private TMP_Text statusText;

    [Header("UI References - Lobby")]
    [SerializeField] private GameObject connectedPanel;
    [SerializeField] private Transform playersListContent;
    [SerializeField] private GameObject playerListItemPrefab; // Simple Text prefab
    [SerializeField] private Button startChatButton;

    [Header("Settings")]
    [SerializeField] private string chatSceneName = "ChatScene";
    [SerializeField] private ushort port = 7777;

    private NetworkManager networkManager;
    private UnityTransport transport;
    private Dictionary<ulong, GameObject> playerListItems = new Dictionary<ulong, GameObject>();

    private void Awake()
    {
        DontDestroyOnLoad(gameObject);
    }

    private void Start()
    {
        networkManager = NetworkManager.Singleton;
        transport = networkManager.GetComponent<UnityTransport>();

        // Configurar UI inicial
        ipAddressInput.text = "127.0.0.1";
        connectedPanel.SetActive(false);
        
        // Añadir listeners a botones
        hostButton.onClick.AddListener(StartHost);
        joinButton.onClick.AddListener(StartClient);
        startChatButton.onClick.AddListener(LoadChatScene);

        statusText.text = $"Your IP: {GetLocalIPAddress()}";

        // Suscribirse a eventos de red
        networkManager.OnClientConnectedCallback += OnClientConnected;
        networkManager.OnClientDisconnectCallback += OnClientDisconnected;
    }

    private void StartHost()
    {
        transport.SetConnectionData("0.0.0.0", port);

        if (networkManager.StartHost())
        {
            statusText.text = $"Hosting on IP: {GetLocalIPAddress()}:{port}";
            Debug.Log("Started as Host");
            
            // Mostrar panel de jugadores conectados
            ShowConnectedPanel();
            
            // Agregar el host a la lista
            AddPlayerToList(networkManager.LocalClientId, "You (Host)");
        }
        else
        {
            statusText.text = "Failed to start host!";
            Debug.LogError("Failed to start host");
        }
    }

    private void StartClient()
    {
        string ipAddress = ipAddressInput.text.Trim();

        if (string.IsNullOrEmpty(ipAddress))
        {
            statusText.text = "Please enter an IP address!";
            return;
        }

        transport.SetConnectionData(ipAddress, port);

        if (networkManager.StartClient())
        {
            statusText.text = $"Connecting to {ipAddress}:{port}...";
            Debug.Log($"Attempting to connect to {ipAddress}:{port}");
        }
        else
        {
            statusText.text = "Failed to start client!";
            Debug.LogError("Failed to start client");
        }
    }

    private void OnClientConnected(ulong clientId)
    {
        Debug.Log($"Client {clientId} connected!");
        
        // Si es el cliente local que se conectó
        if (clientId == networkManager.LocalClientId)
        {
            statusText.text = "Connected successfully!";
            ShowConnectedPanel();
            
            // Si no es el host, agregar "You" a la lista
            if (!networkManager.IsHost)
            {
                AddPlayerToList(clientId, "You");
            }
        }
        else
        {
            // Otro jugador se conectó
            AddPlayerToList(clientId, $"Player {clientId}");
        }
    }

    private void OnClientDisconnected(ulong clientId)
    {
        Debug.Log($"Client {clientId} disconnected");
        
        // Remover de la lista
        RemovePlayerFromList(clientId);
        
        // Si nosotros nos desconectamos
        if (clientId == networkManager.LocalClientId)
        {
            statusText.text = "Disconnected from server";
            HideConnectedPanel();
        }
    }

    private void ShowConnectedPanel()
    {
        // Ocultar controles de conexión
        hostButton.gameObject.SetActive(false);
        joinButton.gameObject.SetActive(false);
        ipAddressInput.gameObject.SetActive(false);
        
        // Mostrar panel de jugadores
        connectedPanel.SetActive(true);
        
        // Solo el host puede iniciar el chat
        startChatButton.gameObject.SetActive(networkManager.IsHost);
    }

    private void HideConnectedPanel()
    {
        // Mostrar controles de conexión
        hostButton.gameObject.SetActive(true);
        joinButton.gameObject.SetActive(true);
        ipAddressInput.gameObject.SetActive(true);
        
        // Ocultar panel de jugadores
        connectedPanel.SetActive(false);
        
        // Limpiar lista
        foreach (var item in playerListItems.Values)
        {
            Destroy(item);
        }
        playerListItems.Clear();
    }

    private void AddPlayerToList(ulong clientId, string playerName)
    {
        // No agregar duplicados
        if (playerListItems.ContainsKey(clientId))
            return;

        // Crear nuevo item en la lista
        GameObject listItem = Instantiate(playerListItemPrefab, playersListContent);
        TMP_Text text = listItem.GetComponent<TMP_Text>();
        
        if (text != null)
        {
            text.text = $"• {playerName}";
        }

        playerListItems.Add(clientId, listItem);
        
        Debug.Log($"Added {playerName} to player list");
    }

    private void RemovePlayerFromList(ulong clientId)
    {
        if (playerListItems.ContainsKey(clientId))
        {
            Destroy(playerListItems[clientId]);
            playerListItems.Remove(clientId);
        }
    }

    private void LoadChatScene()
    {
        if (networkManager.IsHost)
        {
            // El host carga la escena para todos
            networkManager.SceneManager.LoadScene(chatSceneName, LoadSceneMode.Single);
        }
    }

    private string GetLocalIPAddress()
    {
        var host = System.Net.Dns.GetHostEntry(System.Net.Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
            {
                return ip.ToString();
            }
        }
        return "No IP Found";
    }

    private void OnDestroy()
    {
        if (networkManager != null)
        {
            networkManager.OnClientConnectedCallback -= OnClientConnected;
            networkManager.OnClientDisconnectCallback -= OnClientDisconnected;
        }
    }
}