using UnityEngine;
using TMPro;
using Unity.Netcode;
using System.Collections;

public class RoleDisplay : MonoBehaviour
{
    [Header("Text")]
    [SerializeField] private TMP_Text roleText;

    private IEnumerator Start()
    {
        yield return new WaitUntil(() => NetworkManager.Singleton != null && NetworkManager.Singleton.IsConnectedClient);

        SimpleNetworkManager netManager = FindObjectOfType<SimpleNetworkManager>();
        if (netManager == null)
        {
            Debug.LogError("No se encontró el SimpleNetworkManager en la escena.");
            yield break;
        }

        Debug.Log($"ID: {netManager.impostorClientId.Value}");

        yield return new WaitForSeconds(0.5f);

        ulong localClientId = NetworkManager.Singleton.LocalClientId;
        if (netManager.impostorClientId.Value == ulong.MaxValue)
        {
            roleText.text = "Eres un tripulante.";
            roleText.color = Color.green;
        }
        else if (localClientId == netManager.impostorClientId.Value)
        {
            roleText.text = "¡Eres el impostor!";
            roleText.color = Color.red;
        }
        else
        {
            roleText.text = "Eres un tripulante.";
            roleText.color = Color.green;
        }
    }
}
