using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenuManager : MonoBehaviour
{
    // Call this method when Play button is clicked
    public void PlayGame()
    {
        // Load your first game scene (change "GameScene" to your scene name)
        SceneManager.LoadScene("ConnectionMenu");
    }

    // Call this method when Options button is clicked
    public void HowToPlayScreen()
    {
        // You can load an options scene or open an options panel
        Debug.Log("Options clicked - implement your options menu here");
        SceneManager.LoadScene("HowToScene");
    }

    // Call this method when Quit button is clicked
    public void QuitGame()
    {
        Debug.Log("Quit clicked");
        Application.Quit();
        
        // For testing in editor
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #endif
    }
}