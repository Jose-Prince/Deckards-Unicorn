using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using Unity.Services.Core;
using Unity.Services.Authentication;
using Unity.Services.Relay;
using Unity.Services.Relay.Models;

public class SimpleNetworkManager : MonoBehaviour
{
    [Header("UI References - Connection")]
    [SerializeField] private Button hostButton;
    [SerializeField] private Button joinButton;
    [SerializeField] private TMP_InputField ipAddressInput;
    [SerializeField] private TMP_Text statusText;

    [Header("UI References - Lobby")]
    [SerializeField] private Transform playersListContent;
    [SerializeField] private GameObject playerListItemPrefab;
    [SerializeField] private Button startChatButton;

    [Header("Settings")]
    [SerializeField] private string chatSceneName = "ChatScene";

    [SerializeField] private Button returnButton;
    [SerializeField] private Button leaveButton;

    private NetworkManager networkManager;
    private UnityTransport transport;
    private Dictionary<ulong, GameObject> playerListItems = new();

    private void Awake()
    {
        DontDestroyOnLoad(gameObject);
    }

    private async void Start()
    {
        networkManager = NetworkManager.Singleton;
        transport = networkManager.GetComponent<UnityTransport>();

        await UnityServices.InitializeAsync();
        if (!AuthenticationService.Instance.IsSignedIn)
        {
            await AuthenticationService.Instance.SignInAnonymouslyAsync();
        }

        statusText.text = $"Connected to Unity Services.";

        hostButton.onClick.AddListener(StartHost);
        joinButton.onClick.AddListener(StartClient);
        startChatButton.onClick.AddListener(LoadChatScene);

        networkManager.OnClientConnectedCallback += OnClientConnected;
        networkManager.OnClientDisconnectCallback += OnClientDisconnected;

        returnButton.onClick.AddListener(() => SceneManager.LoadScene("MainMenu"));
        leaveButton.onClick.AddListener(LeaveGame);
        leaveButton.gameObject.SetActive(false);
    }

    private async void StartHost()
    {
        try
        {
            statusText.text = "Creating Relay allocation...";

            // 👇 CORRECTO
            Allocation allocation = await RelayService.Instance.CreateAllocationAsync(8);

            string joinCode = await RelayService.Instance.GetJoinCodeAsync(allocation.AllocationId);
            statusText.text = $"Room Code: {joinCode}";
            Debug.Log($"Relay room created. Join code: {joinCode}");

            transport.SetRelayServerData(
                allocation.RelayServer.IpV4,
                (ushort)allocation.RelayServer.Port,
                allocation.AllocationIdBytes,
                allocation.Key,
                allocation.ConnectionData
            );

            if (networkManager.StartHost())
            {
                ShowConnectedPanel();
                AddPlayerToList(networkManager.LocalClientId, "You (Host)");
            }
        }
        catch (RelayServiceException e)
        {
            statusText.text = "Relay Host Error: " + e.Message;
            Debug.LogError(e);
        }
    }

    private async void StartClient()
    {
        string joinCode = ipAddressInput.text.Trim();

        if (string.IsNullOrEmpty(joinCode))
        {
            statusText.text = "Enter a room code!";
            return;
        }

        try
        {
            statusText.text = "Joining Relay...";

            JoinAllocation joinAlloc = await RelayService.Instance.JoinAllocationAsync(joinCode);

            transport.SetRelayServerData(
                joinAlloc.RelayServer.IpV4,
                (ushort)joinAlloc.RelayServer.Port,
                joinAlloc.AllocationIdBytes,
                joinAlloc.Key,
                joinAlloc.ConnectionData,
                joinAlloc.HostConnectionData
            );

            if (networkManager.StartClient())
            {
                statusText.text = $"Joined room {joinCode}";
            }
        }
        catch (RelayServiceException e)
        {
            statusText.text = "Join failed: " + e.Message;
            Debug.LogError(e);
        }
    }

    private void OnClientConnected(ulong clientId)
    {
        Debug.Log($"Client {clientId} connected!");

        if (clientId == networkManager.LocalClientId)
        {
            statusText.text = "Connected successfully!";
            ShowConnectedPanel();

            if (!networkManager.IsHost)
            {
                AddPlayerToList(clientId, "You");
            }
        }
        else
        {
            AddPlayerToList(clientId, $"Player {clientId}");
        }
    }

    private void OnClientDisconnected(ulong clientId)
    {
        Debug.Log($"Client {clientId} disconnected");

        RemovePlayerFromList(clientId);

        if (clientId == networkManager.LocalClientId)
        {
            statusText.text = "Disconnected from server";
            HideConnectedPanel();
        }
    }

    private void ShowConnectedPanel()
    {
        hostButton.gameObject.SetActive(false);
        joinButton.gameObject.SetActive(false);
        ipAddressInput.gameObject.SetActive(false);
        leaveButton.gameObject.SetActive(true);

        startChatButton.gameObject.SetActive(networkManager.IsHost);
    }

    private void HideConnectedPanel()
    {
        hostButton.gameObject.SetActive(true);
        joinButton.gameObject.SetActive(true);
        ipAddressInput.gameObject.SetActive(true);
        leaveButton.gameObject.SetActive(false);

        foreach (var item in playerListItems.Values)
        {
            Destroy(item);
        }
        playerListItems.Clear();
    }

    private void AddPlayerToList(ulong clientId, string playerName)
    {
        if (playerListItems.ContainsKey(clientId)) return;

        GameObject listItem = Instantiate(playerListItemPrefab, playersListContent);
        TMP_Text text = listItem.GetComponentInChildren<TMP_Text>();
        if (text != null) text.text = $" {playerName}";

        playerListItems.Add(clientId, listItem);
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
            networkManager.SceneManager.LoadScene(chatSceneName, LoadSceneMode.Single);
        }
    }

    private void OnDestroy()
    {
        if (networkManager != null)
        {
            networkManager.OnClientConnectedCallback -= OnClientConnected;
            networkManager.OnClientDisconnectCallback -= OnClientDisconnected;
        }
    }

    private void LeaveGame()
    {
        if (networkManager == null) return;

        networkManager.Shutdown();
        HideConnectedPanel();
    }
}
